from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import time
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Hugging Face models...")
print("This may take a minute on first run...")

category_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

event_categories = [
    "systematic issue affecting multiple customers due to bank system or process error requiring remediation",
    "isolated one-time error affecting single customer that was already resolved",
    "needs investigation to determine if systematic or isolated issue"
]

product_categories = ["deposits", "loans", "credit card", "general banking"]
severity_categories = ["high severity", "medium severity", "low severity"]

print("Models loaded successfully!")


def classify_single_event(event):
    event_result = category_classifier(event, candidate_labels=event_categories)
    event_type = event_result['labels'][0]
    event_score = round(event_result['scores'][0] * 100, 1)

    product_result = category_classifier(event, candidate_labels=product_categories)
    product_type = product_result['labels'][0]
    product_score = round(product_result['scores'][0] * 100, 1)

    severity_result = category_classifier(event, candidate_labels=severity_categories)
    severity_type = severity_result['labels'][0]
    severity_score = round(severity_result['scores'][0] * 100, 1)

    if "systematic" in event_type and severity_type == "high severity":
        action = "ESCALATE TO RISK TEAM IMMEDIATELY"
    elif "systematic" in event_type:
        action = "OPEN REMEDIATION WORKSTREAM"
    elif "investigation" in event_type:
        action = "ASSIGN TO SENIOR ANALYST"
    else:
        action = "LOG AND CLOSE"

    if "systematic" in event_type:
        classification = "REMEDIATION EVENT"
    elif "investigation" in event_type:
        classification = "NEEDS INVESTIGATION"
    else:
        classification = "ONE-TIME ERROR"

    return {
        "classification": classification,
        "classification_score": event_score,
        "product": product_type.upper(),
        "product_score": product_score,
        "severity": severity_type.upper(),
        "severity_score": severity_score,
        "action": action
    }


@app.route("/")
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Bharat Upadhyay's Remediation AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; color: #1a1a1a; min-height: 100vh; padding: 40px 20px; }
        .app { max-width: 1000px; margin: 0 auto; }
        .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding-bottom: 1.25rem; border-bottom: 0.5px solid #e0e0e0; }
        .logo { width: 48px; height: 48px; background: #7F77DD; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 24px; }
        .brand { font-size: 26px; font-weight: 700; color: #1a1a1a; }
        .brand span { color: #7F77DD; font-size: 28px; font-weight: 800; }
        .badge { margin-left: auto; font-size: 11px; background: #EEEDFE; color: #3C3489; padding: 4px 12px; border-radius: 20px; font-weight: 500; border: 0.5px solid #7F77DD; }
        .tabs { display: flex; margin-bottom: 1.5rem; border-bottom: 0.5px solid #e0e0e0; }
        .tab { padding: 10px 24px; font-size: 14px; cursor: pointer; border-bottom: 2px solid transparent; color: #888888; font-weight: 500; transition: all 0.15s; }
        .tab.active { color: #7F77DD; border-bottom-color: #7F77DD; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 1.5rem; }
        .stat { background: #ffffff; border: 0.5px solid #e0e0e0; border-radius: 10px; padding: 14px 16px; }
        .stat-val { font-size: 22px; font-weight: 500; color: #1a1a1a; }
        .stat-label { font-size: 12px; color: #888888; margin-top: 2px; }
        .input-card { background: #ffffff; border: 0.5px solid #e0e0e0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
        .input-label { font-size: 11px; color: #888888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.06em; }
        textarea { width: 100%; background: #f5f5f5; border: 0.5px solid #dddddd; border-radius: 8px; padding: 12px 14px; font-size: 14px; color: #1a1a1a; resize: vertical; font-family: inherit; min-height: 100px; outline: none; transition: border-color 0.15s; }
        textarea:focus { border-color: #7F77DD; }
        .actions { display: flex; gap: 10px; margin-top: 12px; }
        .btn-primary { background: #7F77DD; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer; }
        .btn-primary:disabled { background: #c5c2f0; cursor: not-allowed; }
        .btn-secondary { background: transparent; color: #777777; border: 0.5px solid #dddddd; border-radius: 8px; padding: 10px 20px; font-size: 14px; cursor: pointer; }
        .btn-download { background: #1D9E75; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer; display: none; margin-bottom: 1rem; }
        .loading { display: none; align-items: center; gap: 8px; padding: 12px 0; font-size: 13px; color: #888888; }
        .spinner { width: 14px; height: 14px; border: 2px solid #e0e0e0; border-top-color: #7F77DD; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result-card { background: #ffffff; border: 0.5px solid #e0e0e0; border-radius: 12px; overflow: hidden; margin-bottom: 1rem; display: none; }
        .result-header { display: flex; align-items: center; padding: 10px 16px; border-bottom: 0.5px solid #eeeeee; background: #fafafa; }
        .result-title { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 0.06em; display: flex; align-items: center; gap: 6px; }
        .dot { width: 7px; height: 7px; border-radius: 50%; background: #7F77DD; }
        .result-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; background: #eeeeee; }
        .result-item { background: #ffffff; padding: 16px 20px; }
        .result-item-label { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
        .result-item-value { font-size: 16px; font-weight: 500; color: #1a1a1a; }
        .result-item-score { font-size: 12px; color: #888888; margin-top: 2px; }
        .action-bar { padding: 14px 20px; display: flex; align-items: center; gap: 10px; }
        .action-label { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 0.06em; }
        .action-pill { font-size: 13px; font-weight: 600; padding: 6px 16px; border-radius: 20px; }
        .mini-pill { font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; white-space: nowrap; }
        .action-escalate { background: #FCEBEB; color: #A32D2D; }
        .action-workstream { background: #FAEEDA; color: #633806; }
        .action-investigate { background: #E6F1FB; color: #0C447C; }
        .action-log { background: #E1F5EE; color: #085041; }
        .samples { margin-bottom: 1.5rem; }
        .samples-label { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
        .sample-btn { display: inline-block; font-size: 12px; background: #ffffff; color: #7F77DD; padding: 6px 14px; border-radius: 20px; margin: 3px; border: 0.5px solid #7F77DD; cursor: pointer; }
        .sample-btn:hover { background: #EEEDFE; }
        .upload-zone { border: 2px dashed #7F77DD; border-radius: 12px; padding: 3rem; text-align: center; background: #FAFAFE; cursor: pointer; margin-bottom: 1rem; }
        .upload-zone:hover { background: #EEEDFE; }
        .upload-icon { font-size: 48px; margin-bottom: 12px; }
        .upload-title { font-size: 18px; font-weight: 500; color: #1a1a1a; margin-bottom: 6px; }
        .upload-subtitle { font-size: 13px; color: #888888; }
        .batch-results { background: #ffffff; border: 0.5px solid #e0e0e0; border-radius: 12px; overflow: hidden; display: none; margin-bottom: 1rem; }
        .batch-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; background: #fafafa; border-bottom: 0.5px solid #eeeeee; }
        .batch-title { font-size: 14px; font-weight: 500; color: #1a1a1a; }
        .batch-count { font-size: 12px; color: #888888; }
        .batch-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .batch-table th { background: #f5f5f5; padding: 10px 14px; text-align: left; font-size: 11px; font-weight: 500; color: #888888; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 0.5px solid #eeeeee; }
        .batch-table td { padding: 12px 14px; border-bottom: 0.5px solid #f5f5f5; color: #1a1a1a; vertical-align: top; }
        .progress-bar { display: none; margin: 1rem 0; }
        .progress-label { font-size: 13px; color: #888888; margin-bottom: 6px; }
        .progress-track { background: #eeeeee; border-radius: 20px; height: 8px; overflow: hidden; }
        .progress-fill { background: #7F77DD; height: 100%; border-radius: 20px; transition: width 0.3s; width: 0%; }
        .history-section { margin-top: 2rem; }
        .section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
        .section-title { font-size: 11px; font-weight: 500; color: #888888; text-transform: uppercase; letter-spacing: 0.06em; }
        .clear-link { font-size: 12px; color: #c0392b; cursor: pointer; background: none; border: none; }
        .history-item { background: #ffffff; border: 0.5px solid #eeeeee; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; cursor: pointer; transition: border-color 0.15s; }
        .history-item:hover { border-color: #7F77DD; }
        .h-event { font-size: 13px; color: #666666; margin-bottom: 4px; }
        .h-result { font-size: 12px; color: #7F77DD; font-weight: 500; }
        .h-time { font-size: 11px; color: #999999; margin-top: 4px; }
        .no-history { font-size: 13px; color: #aaaaaa; text-align: center; padding: 2rem 0; }
    </style>
</head>
<body>
<div class="app">

    <div class="topbar">
        <div class="logo">👑</div>
        <div><div class="brand"><span>Bharat Upadhyay\'s</span> Remediation AI</div></div>
        <div class="badge">Powered by Hugging Face</div>
    </div>

    <div class="stats">
        <div class="stat"><div class="stat-val" id="stat-count">0</div><div class="stat-label">Events classified</div></div>
        <div class="stat"><div class="stat-val" id="stat-remediation">0</div><div class="stat-label">Remediation events</div></div>
        <div class="stat"><div class="stat-val" id="stat-escalate">0</div><div class="stat-label">Escalations raised</div></div>
        <div class="stat"><div class="stat-val">100%</div><div class="stat-label">Local and private</div></div>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="switchTab(event, \'single\')">Single Event</div>
        <div class="tab" onclick="switchTab(event, \'batch\')">Batch Excel Upload</div>
    </div>

    <div class="tab-content active" id="tab-single">
        <div class="samples">
            <div class="samples-label">Try these events</div>
            <span class="sample-btn" onclick="loadSample(this)">System upgrade caused bonus interest flag to be overridden for all deposit accounts</span>
            <span class="sample-btn" onclick="loadSample(this)">Single customer BSB error corrected same day by branch staff</span>
            <span class="sample-btn" onclick="loadSample(this)">COVID relief loan fees charged to 28,000 customers despite fee waiver in T&Cs</span>
        </div>

        <div class="input-card">
            <div class="input-label">Describe the remediation event</div>
            <textarea id="event" placeholder="e.g. System upgrade caused bonus interest flag to be overridden..."></textarea>
            <div class="actions">
                <button class="btn-primary" id="classifyBtn" onclick="classifyEvent()">👑 Classify Event</button>
                <button class="btn-secondary" onclick="clearInput()">Clear</button>
            </div>
            <div class="loading" id="loading"><div class="spinner"></div>Analysing event...</div>
        </div>

        <div class="result-card" id="resultCard">
            <div class="result-header"><div class="result-title"><div class="dot"></div>Classification Result</div></div>
            <div class="result-grid">
                <div class="result-item"><div class="result-item-label">Classification</div><div class="result-item-value" id="res-classification">—</div><div class="result-item-score" id="res-classification-score"></div></div>
                <div class="result-item"><div class="result-item-label">Product</div><div class="result-item-value" id="res-product">—</div><div class="result-item-score" id="res-product-score"></div></div>
                <div class="result-item"><div class="result-item-label">Severity</div><div class="result-item-value" id="res-severity">—</div><div class="result-item-score" id="res-severity-score"></div></div>
                <div class="result-item"><div class="result-item-label">Recommended action</div><div class="result-item-value" id="res-action">—</div></div>
            </div>
            <div class="action-bar">
                <div class="action-label">Action required:</div>
                <div class="action-pill" id="res-action-pill">—</div>
            </div>
        </div>

        <div class="history-section">
            <div class="section-header">
                <div class="section-title">Event history</div>
                <button class="clear-link" onclick="clearHistory()">Clear all</button>
            </div>
            <div id="history"></div>
        </div>
    </div>

    <div class="tab-content" id="tab-batch">
        <div class="upload-zone" onclick="document.getElementById(\'fileInput\').click()">
            <div class="upload-icon">📊</div>
            <div class="upload-title">Upload your Excel file</div>
            <div class="upload-subtitle">Click to browse — must have an "Event Description" column</div>
            <input type="file" id="fileInput" accept=".xlsx,.xls" style="display:none" onchange="uploadFile(this)">
        </div>

        <div class="progress-bar" id="progressBar">
            <div class="progress-label" id="progressLabel">Processing events...</div>
            <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
        </div>

        <button class="btn-download" id="downloadBtn" onclick="downloadResults()">📥 Download Results Excel</button>

        <div class="batch-results" id="batchResults">
            <div class="batch-header">
                <div class="batch-title">Classification Results</div>
                <div class="batch-count" id="batchCount"></div>
            </div>
            <div style="overflow-x:auto;">
                <table class="batch-table">
                    <thead>
                        <tr>
                            <th>Event ID</th>
                            <th>Event Description</th>
                            <th>Classification</th>
                            <th>Product</th>
                            <th>Severity</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="batchTableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

</div>

<script>
    let totalCount = 0;
    let remediationCount = 0;
    let escalateCount = 0;
    let historyItems = [];

    function switchTab(e, tab) {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
        e.target.classList.add("active");
        document.getElementById("tab-" + tab).classList.add("active");
    }

    async function classifyEvent() {
        const eventText = document.getElementById("event").value.trim();
        if (!eventText) return;
        document.getElementById("classifyBtn").disabled = true;
        document.getElementById("loading").style.display = "flex";
        document.getElementById("resultCard").style.display = "none";
        const response = await fetch("/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ event: eventText }) });
        const data = await response.json();
        document.getElementById("classifyBtn").disabled = false;
        document.getElementById("loading").style.display = "none";
        document.getElementById("resultCard").style.display = "block";
        document.getElementById("res-classification").innerText = data.classification;
        document.getElementById("res-classification-score").innerText = data.classification_score + "% confidence";
        document.getElementById("res-product").innerText = data.product;
        document.getElementById("res-product-score").innerText = data.product_score + "% confidence";
        document.getElementById("res-severity").innerText = data.severity;
        document.getElementById("res-severity-score").innerText = data.severity_score + "% confidence";
        document.getElementById("res-action").innerText = data.action;
        const pill = document.getElementById("res-action-pill");
        pill.innerText = data.action;
        pill.className = "action-pill";
        if (data.action.includes("ESCALATE")) pill.classList.add("action-escalate");
        else if (data.action.includes("WORKSTREAM")) pill.classList.add("action-workstream");
        else if (data.action.includes("ANALYST")) pill.classList.add("action-investigate");
        else pill.classList.add("action-log");
        totalCount++;
        document.getElementById("stat-count").innerText = totalCount;
        if (data.classification === "REMEDIATION EVENT") { remediationCount++; document.getElementById("stat-remediation").innerText = remediationCount; }
        if (data.action.includes("ESCALATE")) { escalateCount++; document.getElementById("stat-escalate").innerText = escalateCount; }
        const now = new Date();
        historyItems.unshift({ event: eventText, classification: data.classification, action: data.action, time: now.toLocaleTimeString() });
        renderHistory();
    }

    function loadSample(el) { document.getElementById("event").value = el.innerText; classifyEvent(); }
    function clearInput() { document.getElementById("event").value = ""; document.getElementById("resultCard").style.display = "none"; }

    function renderHistory() {
        const container = document.getElementById("history");
        if (historyItems.length === 0) { container.innerHTML = "<div class=\'no-history\'>No events classified yet.</div>"; return; }
        container.innerHTML = historyItems.map((item, i) => `<div class="history-item" onclick="loadHistoryItem(${i})"><div class="h-event">📋 ${item.event.substring(0, 80)}...</div><div class="h-result">${item.classification} → ${item.action}</div><div class="h-time">🕐 ${item.time}</div></div>`).join("");
    }

    function loadHistoryItem(i) { document.getElementById("event").value = historyItems[i].event; classifyEvent(); }
    function clearHistory() { historyItems = []; renderHistory(); }

    async function uploadFile(input) {
        const file = input.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append("file", file);
        document.getElementById("progressBar").style.display = "block";
        document.getElementById("progressFill").style.width = "10%";
        document.getElementById("progressLabel").innerText = "Uploading and classifying all events...";
        document.getElementById("batchResults").style.display = "none";
        document.getElementById("downloadBtn").style.display = "none";
        const response = await fetch("/batch_classify", { method: "POST", body: formData });
        const data = await response.json();
        document.getElementById("progressFill").style.width = "100%";
        document.getElementById("progressLabel").innerText = "Done! " + data.count + " events classified.";
        if (data.error) { alert("Error: " + data.error); return; }
        renderBatchResults(data.results);
        totalCount += data.results.length;
        document.getElementById("stat-count").innerText = totalCount;
        const remCount = data.results.filter(r => r.classification === "REMEDIATION EVENT").length;
        const escCount = data.results.filter(r => r.action.includes("ESCALATE")).length;
        remediationCount += remCount; escalateCount += escCount;
        document.getElementById("stat-remediation").innerText = remediationCount;
        document.getElementById("stat-escalate").innerText = escalateCount;
    }

    function renderBatchResults(results) {
        document.getElementById("batchResults").style.display = "block";
        document.getElementById("downloadBtn").style.display = "block";
        document.getElementById("batchCount").innerText = results.length + " events classified";
        document.getElementById("batchTableBody").innerHTML = results.map(r => {
            let pc = r.action.includes("ESCALATE") ? "action-escalate" : r.action.includes("WORKSTREAM") ? "action-workstream" : r.action.includes("ANALYST") ? "action-investigate" : "action-log";
            return `<tr><td style="font-family:monospace;font-size:12px;color:#888">${r.event_id}</td><td style="max-width:280px;font-size:12px">${r.event.substring(0, 80)}...</td><td><span class="mini-pill ${pc}">${r.classification}</span></td><td style="font-size:12px">${r.product}</td><td style="font-size:12px">${r.severity}</td><td><span class="mini-pill ${pc}">${r.action}</span></td></tr>`;
        }).join("");
    }

    function downloadResults() { window.location.href = "/download_results"; }
    renderHistory();
</script>
</body>
</html>
'''


@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    event = data.get("event", "")
    result = classify_single_event(event)
    return jsonify(result)


@app.route("/batch_classify", methods=["POST"])
def batch_classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    try:
        df = pd.read_excel(filepath)
        if "Event Description" not in df.columns:
            return jsonify({"error": "Excel must have an 'Event Description' column"})
        results = []
        for index, row in df.iterrows():
            event_text = str(row["Event Description"])
            event_id = str(row.get("Event ID", f"EVT-{index+1}"))
            result = classify_single_event(event_text)
            result["event"] = event_text
            result["event_id"] = event_id
            results.append(result)
        results_df = pd.DataFrame(results)
        results_df = results_df.rename(columns={
            "event_id": "Event ID", "event": "Event Description",
            "classification": "Classification", "classification_score": "Classification Confidence %",
            "product": "Product", "product_score": "Product Confidence %",
            "severity": "Severity", "severity_score": "Severity Confidence %",
            "action": "Recommended Action"
        })
        results_path = os.path.join(UPLOAD_FOLDER, "classified_results.xlsx")
        results_df.to_excel(results_path, index=False)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/download_results")
def download_results():
    results_path = os.path.join(UPLOAD_FOLDER, "classified_results.xlsx")
    return send_file(results_path, as_attachment=True, download_name="King_of_Remediation_Results.xlsx")


if __name__ == "__main__":
    app.run(debug=True)
