from flask import Flask, request, jsonify
from transformers import pipeline
import time

app = Flask(__name__)

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

@app.route("/")
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Bharat Upadhyay's King of Remediation AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f5f5;
            color: #1a1a1a;
            min-height: 100vh;
            padding: 40px 20px;
        }

        .app { max-width: 900px; margin: 0 auto; }

        .topbar {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 2rem;
            padding-bottom: 1.25rem;
            border-bottom: 0.5px solid #e0e0e0;
        }

        .logo {
            width: 48px;
            height: 48px;
            background: #7F77DD;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .brand { font-size: 26px; font-weight: 700; color: #1a1a1a; }
        .brand span { color: #7F77DD; font-size: 28px; font-weight: 800; }

        .badge {
            margin-left: auto;
            font-size: 11px;
            background: #EEEDFE;
            color: #3C3489;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 500;
            border: 0.5px solid #7F77DD;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 1.5rem;
        }

        .stat {
            background: #ffffff;
            border: 0.5px solid #e0e0e0;
            border-radius: 10px;
            padding: 14px 16px;
        }

        .stat-val { font-size: 22px; font-weight: 500; color: #1a1a1a; }
        .stat-label { font-size: 12px; color: #888888; margin-top: 2px; }

        .input-card {
            background: #ffffff;
            border: 0.5px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .input-label {
            font-size: 11px;
            color: #888888;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        textarea {
            width: 100%;
            background: #f5f5f5;
            border: 0.5px solid #dddddd;
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 14px;
            color: #1a1a1a;
            resize: vertical;
            font-family: inherit;
            min-height: 100px;
            outline: none;
            transition: border-color 0.15s;
        }

        textarea:focus { border-color: #7F77DD; }

        .actions { display: flex; gap: 10px; margin-top: 12px; }

        .btn-primary {
            background: #7F77DD;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.15s;
        }

        .btn-primary:hover { background: #6560c0; }
        .btn-primary:disabled { background: #c5c2f0; cursor: not-allowed; }

        .btn-secondary {
            background: transparent;
            color: #777777;
            border: 0.5px solid #dddddd;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
        }

        .loading {
            display: none;
            align-items: center;
            gap: 8px;
            padding: 12px 0;
            font-size: 13px;
            color: #888888;
        }

        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid #e0e0e0;
            border-top-color: #7F77DD;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .result-card {
            background: #ffffff;
            border: 0.5px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
            display: none;
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 16px;
            border-bottom: 0.5px solid #eeeeee;
            background: #fafafa;
        }

        .result-title {
            font-size: 11px;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .dot { width: 7px; height: 7px; border-radius: 50%; background: #7F77DD; }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1px;
            background: #eeeeee;
        }

        .result-item {
            background: #ffffff;
            padding: 16px 20px;
        }

        .result-item-label {
            font-size: 11px;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        .result-item-value {
            font-size: 16px;
            font-weight: 500;
            color: #1a1a1a;
        }

        .result-item-score {
            font-size: 12px;
            color: #888888;
            margin-top: 2px;
        }

        .action-bar {
            padding: 14px 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .action-label {
            font-size: 11px;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .action-pill {
            font-size: 13px;
            font-weight: 600;
            padding: 6px 16px;
            border-radius: 20px;
        }

        .action-escalate { background: #FCEBEB; color: #A32D2D; }
        .action-workstream { background: #FAEEDA; color: #633806; }
        .action-investigate { background: #E6F1FB; color: #0C447C; }
        .action-log { background: #E1F5EE; color: #085041; }

        .samples {
            margin-bottom: 1.5rem;
        }

        .samples-label {
            font-size: 11px;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }

        .sample-btn {
            display: inline-block;
            font-size: 12px;
            background: #ffffff;
            color: #7F77DD;
            padding: 6px 14px;
            border-radius: 20px;
            margin: 3px;
            border: 0.5px solid #7F77DD;
            cursor: pointer;
            transition: background 0.15s;
        }

        .sample-btn:hover { background: #EEEDFE; }

        .history-section { margin-top: 2rem; }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .section-title {
            font-size: 11px;
            font-weight: 500;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .clear-link {
            font-size: 12px;
            color: #c0392b;
            cursor: pointer;
            background: none;
            border: none;
        }

        .history-item {
            background: #ffffff;
            border: 0.5px solid #eeeeee;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: border-color 0.15s;
        }

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
        <div>
            <div class="brand"><span>Bharat Upadhyay's</span> King of Remediation AI</div>
        </div>
        <div class="badge">Powered by Hugging Face</div>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-val" id="stat-count">0</div>
            <div class="stat-label">Events classified</div>
        </div>
        <div class="stat">
            <div class="stat-val" id="stat-remediation">0</div>
            <div class="stat-label">Remediation events found</div>
        </div>
        <div class="stat">
            <div class="stat-val">100%</div>
            <div class="stat-label">Local and private</div>
        </div>
    </div>

    <div class="samples">
        <div class="samples-label">Try these events</div>
        <span class="sample-btn" onclick="loadSample(this)">System upgrade caused bonus interest flag to be overridden for all deposit accounts</span>
        <span class="sample-btn" onclick="loadSample(this)">Single customer BSB error corrected same day by branch staff</span>
        <span class="sample-btn" onclick="loadSample(this)">COVID relief loan fees charged to 28,000 customers despite fee waiver in T&Cs</span>
        <span class="sample-btn" onclick="loadSample(this)">Credit card annual fee charged to customers on fee waiver program after system patch</span>
    </div>

    <div class="input-card">
        <div class="input-label">Describe the remediation event</div>
        <textarea id="event" placeholder="e.g. System upgrade in January 2019 caused bonus interest rate flag to be overridden for all deposit accounts enrolled in the bonus program..."></textarea>
        <div class="actions">
            <button class="btn-primary" id="classifyBtn" onclick="classifyEvent()">
                👑 Classify Event
            </button>
            <button class="btn-secondary" onclick="clearInput()">Clear</button>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            Analysing event...
        </div>
    </div>

    <div class="result-card" id="resultCard">
        <div class="result-header">
            <div class="result-title"><div class="dot"></div> Classification Result</div>
        </div>
        <div class="result-grid">
            <div class="result-item">
                <div class="result-item-label">Classification</div>
                <div class="result-item-value" id="res-classification">—</div>
                <div class="result-item-score" id="res-classification-score"></div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Product</div>
                <div class="result-item-value" id="res-product">—</div>
                <div class="result-item-score" id="res-product-score"></div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Severity</div>
                <div class="result-item-value" id="res-severity">—</div>
                <div class="result-item-score" id="res-severity-score"></div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Recommended action</div>
                <div class="result-item-value" id="res-action">—</div>
            </div>
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

<script>
    let totalCount = 0;
    let remediationCount = 0;
    let historyItems = [];

    async function classifyEvent() {
        const event = document.getElementById("event").value.trim();
        if (!event) return;

        document.getElementById("classifyBtn").disabled = true;
        document.getElementById("loading").style.display = "flex";
        document.getElementById("resultCard").style.display = "none";

        const response = await fetch("/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ event })
        });

        const data = await response.json();

        document.getElementById("classifyBtn").disabled = false;
        document.getElementById("loading").style.display = "none";
        document.getElementById("resultCard").style.display = "block";

        document.getElementById("res-classification").innerText = data.classification;
        document.getElementById("res-classification-score").innerText = data.classification_score + "% confidence";
        document.getElementById("res-product").innerText = data.product.toUpperCase();
        document.getElementById("res-product-score").innerText = data.product_score + "% confidence";
        document.getElementById("res-severity").innerText = data.severity.toUpperCase();
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

        if (data.classification.toLowerCase().includes("systematic")) {
            remediationCount++;
            document.getElementById("stat-remediation").innerText = remediationCount;
        }

        const now = new Date();
        historyItems.unshift({
            event: event,
            classification: data.classification,
            action: data.action,
            time: now.toLocaleTimeString()
        });

        renderHistory();
    }

    function loadSample(el) {
        document.getElementById("event").value = el.innerText;
        classifyEvent();
    }

    function clearInput() {
        document.getElementById("event").value = "";
        document.getElementById("resultCard").style.display = "none";
    }

    function renderHistory() {
        const container = document.getElementById("history");
        if (historyItems.length === 0) {
            container.innerHTML = "<div class='no-history'>No events classified yet. Try one above!</div>";
            return;
        }
        container.innerHTML = historyItems.map((item, i) => `
            <div class="history-item" onclick="loadHistory(${i})">
                <div class="h-event">📋 ${item.event.substring(0, 80)}...</div>
                <div class="h-result">${item.classification} → ${item.action}</div>
                <div class="h-time">🕐 ${item.time}</div>
            </div>
        `).join("");
    }

    function loadHistory(index) {
        const item = historyItems[index];
        document.getElementById("event").value = item.event;
        classifyEvent();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function clearHistory() {
        historyItems = [];
        totalCount = 0;
        remediationCount = 0;
        document.getElementById("stat-count").innerText = 0;
        document.getElementById("stat-remediation").innerText = 0;
        renderHistory();
    }

    renderHistory();
</script>
</body>
</html>
'''

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    event = data.get("event", "")

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

    return jsonify({
        "classification": classification,
        "classification_score": event_score,
        "product": product_type,
        "product_score": product_score,
        "severity": severity_type,
        "severity_score": severity_score,
        "action": action
    })

if __name__ == "__main__":
    app.run(debug=True)
