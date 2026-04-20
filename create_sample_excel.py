import pandas as pd

# Sample events — the kind your team receives every Monday
events = [
    "System upgrade in January 2019 caused bonus interest rate flag to be overridden for all deposit accounts enrolled in the bonus program",
    "A single customer direct debit failed due to incorrect BSB entered by the customer and was corrected same day",
    "COVID relief loan establishment fees were charged to 28,000 customers despite Terms and Conditions stating fees would be waived",
    "One customer received duplicate statement due to printing error which was corrected immediately",
    "Credit card annual fees were charged to customers on the fee waiver program after a system patch in July 2020 reset the waiver flag",
    "A teller entered wrong amount for a single cash deposit which was reversed same day",
    "Interest rates on fixed term deposits were not updated after RBA rate change affecting 15,000 customers across 3 product lines",
    "Single customer complaint about branch closing time resolved by branch manager on the same day",
    "Loan repayment amounts were incorrectly calculated for 5,000 customers on variable rate home loans after a system migration",
    "One customer was sent a letter addressed to wrong name due to data entry error corrected within 24 hours"
]

# Create DataFrame — like a spreadsheet in Python
df = pd.DataFrame({
    "Event ID": [f"EVT-2026-{str(i+1).zfill(3)}" for i in range(len(events))],
    "Event Description": events,
    "Date Received": ["01/04/2026"] * len(events),
    "Source": ["Event Management Team"] * len(events)
})

# Save to Excel
df.to_excel("sample_events.xlsx", index=False)
print(f"Created sample_events.xlsx with {len(events)} events!")
print("Open it in Excel to see how it looks.")
