from transformers import pipeline

print("Loading models...")

sentiment_classifier = pipeline("sentiment-analysis")

category_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Real banking remediation events
# These are the kind of events your team receives
events = [
    "System upgrade in January 2019 caused bonus interest rate flag to be overridden for all deposit accounts enrolled in the bonus program",
    "A single customer's direct debit failed due to incorrect BSB entered by the customer",
    "COVID relief loan establishment fees were charged to 28,000 customers despite Terms and Conditions stating fees would be waived",
    "One customer received duplicate statement due to printing error which was corrected immediately",
    "Credit card annual fees were charged to customers on the fee waiver program after a system patch in July 2020 reset the waiver flag",
    "A teller entered wrong amount for a single cash deposit which was reversed same day",
    "Interest rates on fixed term deposits were not updated after RBA rate change affecting 15,000 customers across 3 product lines",
    "Single customer complaint about branch closing time — resolved by branch manager"
]

# Remediation classification labels — detailed descriptions improve accuracy
event_categories = [
    "systematic issue affecting multiple customers due to bank system or process error requiring remediation",
    "isolated one-time error affecting single customer that was already resolved",
    "needs investigation to determine if systematic or isolated issue"
]

product_categories = ["deposits", "loans", "credit card", "general banking"]

severity_categories = ["high severity", "medium severity", "low severity"]

print("\n" + "="*70)
print("BHARAT UPADHYAY'S REMEDIATION EVENT CLASSIFIER")
print("="*70)

for event in events:

    # Classify event type
    event_result = category_classifier(
        event,
        candidate_labels=event_categories
    )
    event_type = event_result['labels'][0]
    event_score = round(event_result['scores'][0] * 100, 1)

    # Classify product
    product_result = category_classifier(
        event,
        candidate_labels=product_categories
    )
    product_type = product_result['labels'][0]
    product_score = round(product_result['scores'][0] * 100, 1)

    # Classify severity
    severity_result = category_classifier(
        event,
        candidate_labels=severity_categories
    )
    severity_type = severity_result['labels'][0]
    severity_score = round(severity_result['scores'][0] * 100, 1)

    # Business logic — what action to take
    # Uses 'in' to check for keywords in the longer label descriptions
    if "systematic" in event_type and severity_type == "high severity":
        action = "ESCALATE TO RISK TEAM IMMEDIATELY"
    elif "systematic" in event_type:
        action = "OPEN REMEDIATION WORKSTREAM"
    elif "investigation" in event_type:
        action = "ASSIGN TO SENIOR ANALYST"
    else:
        action = "LOG AND CLOSE"

    # Display emoji based on event type
    if "systematic" in event_type:
        event_emoji = "🚨"
    elif "investigation" in event_type:
        event_emoji = "⚠️"
    else:
        event_emoji = "✅"

    print(f"\n{event_emoji} Event: '{event[:65]}...'")
    print(f"   Classification: {event_type.upper()} ({event_score}%)")
    print(f"   Product:        {product_type.upper()} ({product_score}%)")
    print(f"   Severity:       {severity_type.upper()} ({severity_score}%)")
    print(f"   Action:         {action}")
    print(f"   {'-'*60}")
