# 🚨 Bharat Upadhyay's Remediation Event Classifier

An AI classifier that automatically analyses banking events and determines 
whether they require remediation, are one-time errors, or need investigation.

No cloud. No API costs. Runs 100% on your machine.

---

## 💡 The Problem

Banking remediation teams receive dozens of events daily. Business analysts 
manually read each one and spend hours deciding:
- Is this a systematic remediation event or a one-time error?
- Which product is affected — deposits, loans or credit card?
- How severe is it?
- What action should be taken?

This tool does all of that in milliseconds with confidence scores.

---

## ✨ What it does

Feed it a banking event description and it instantly returns:
- Classification — remediation event, one-time error, or needs investigation
- Product — deposits, loans, credit card, or general banking
- Severity — high, medium, or low
- Action — escalate, open workstream, assign to analyst, or log and close

---

## 🏦 Real World Example

**Input:**
> "COVID relief loan establishment fees were charged to 28,000 customers 
> despite Terms and Conditions stating fees would be waived"

**Output:**
