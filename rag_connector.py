from sentence_transformers import SentenceTransformer
import chromadb
import requests
import sys
import os

# Point to the Confluence RAG Agent's ChromaDB
# We built this in Week 2 — reusing it here!
CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../confluence-rag-agent/chroma_db"
)

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Confluence ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="remediation_docs",
    metadata={"hnsw:space": "cosine"}
)

print(f"Connected! {collection.count()} Confluence pages available.")


def find_similar_cases(event_description, n_results=3):
    """Find similar past Confluence cases for this event"""

    # Convert event to fingerprint
    question_embedding = embedding_model.encode(
        event_description
    ).tolist()

    # Search ChromaDB for similar pages
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    similar_cases = []
    context = ""

    for i, (doc, metadata) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0]
    )):
        similar_cases.append({
            "title": metadata['title'],
            "category": metadata['category'],
            "product": metadata['product'],
            "content": doc[:500]  # first 500 chars for preview
        })
        context += f"\n\nCONFLUENCE PAGE {i+1}: {metadata['title']}\n{doc}"

    return similar_cases, context


def check_for_conflicts(hf_classification, similar_cases):
    """Check if RAG results contradict Hugging Face classification"""

    conflict_keywords = [
        "not remediation",
        "ruled out",
        "one-time",
        "no remediation required",
        "excluded",
        "not applicable"
    ]

    for case in similar_cases:
        content_lower = case['content'].lower()
        for keyword in conflict_keywords:
            if keyword in content_lower:
                if hf_classification == "REMEDIATION EVENT":
                    return {
                        "conflict": True,
                        "message": f"Conflicting signal found in: {case['title']}",
                        "conflicting_case": case['title']
                    }

    return {"conflict": False}


def get_llama_explanation(event_description, hf_result, similar_cases, context):
    """Ask LLaMA to explain the classification and recommend stages"""

    prompt = f"""You are an expert banking remediation analyst with 14 years of experience.

A new event has been received:
EVENT: {event_description}

AI CLASSIFICATION RESULT:
- Classification: {hf_result['classification']} ({hf_result['classification_score']}% confidence)
- Product: {hf_result['product']}
- Severity: {hf_result['severity']}

SIMILAR PAST CASES FROM CONFLUENCE:
{context}

Based on the classification and similar past cases, please provide:

1. EXPLANATION (2-3 sentences): Why does this event match or not match a remediation event?
2. SIMILAR CASES SUMMARY: What do the past cases tell us about this event?
3. RECOMMENDED STAGES: Which of these stages are needed?
   - Event Details
   - Data Discovery
   - Account Identification
   - Base Refund Calculation
   - Secondary Refund
   - Time Value of Money
4. KEY RISKS: What are the main risks to watch for?

Be concise and specific. Use banking terminology."""

    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


def full_analysis(event_description, hf_result):
    """Complete analysis combining HF + RAG + LLaMA"""

    print(f"  Searching Confluence for similar cases...")
    similar_cases, context = find_similar_cases(event_description)

    print(f"  Checking for conflicts...")
    conflict = check_for_conflicts(hf_result['classification'], similar_cases)

    print(f"  Generating LLaMA explanation...")
    explanation = get_llama_explanation(
        event_description, hf_result, similar_cases, context
    )

    return {
        "similar_cases": similar_cases,
        "conflict": conflict,
        "explanation": explanation
    }
