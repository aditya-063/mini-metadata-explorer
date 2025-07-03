import pandas as pd
import numpy as np
from faker import Faker
import plotly.graph_objects as go

# Seed
fake = Faker()
np.random.seed(42)
Faker.seed(42)

# Generate Datasets
def generate_customer_data(n=300):
    return pd.DataFrame({
        "customer_id": [f"CUST{1000+i}" for i in range(n)],
        "name": [fake.name() for _ in range(n)],
        "email": [fake.email() for _ in range(n)],
        "phone": [fake.phone_number() for _ in range(n)],
        "signup_date": [fake.date_between(start_date="-5y", end_date="today") for _ in range(n)],
        "country": np.random.choice(["USA", "India", "Germany", "Canada"], size=n),
        "age": np.random.randint(18, 65, size=n),
        "is_active": np.random.choice([True, False], size=n),
    })

def generate_transaction_data(n=500):
    return pd.DataFrame({
        "transaction_id": [f"TXN{5000+i}" for i in range(n)],
        "cust_id": [f"CUST{1000 + np.random.randint(0, 300)}" for _ in range(n)],  # intentionally different name
        "transaction_date": [fake.date_between(start_date="-3y", end_date="today") for _ in range(n)],
        "amount": np.round(np.random.uniform(10, 1000, size=n), 2),
        "payment_method": np.random.choice(["Card", "UPI", "Netbanking", "Cash"], size=n),
        "status": np.random.choice(["Completed", "Pending", "Failed"], size=n),
    })

def generate_support_data(n=150):
    return pd.DataFrame({
        "ticket_id": [f"TICKET{8000+i}" for i in range(n)],
        "customer_id": [f"CUST{1000 + np.random.randint(0, 300)}" for _ in range(n)],
        "issue_type": np.random.choice(["Login", "Payment", "Account", "Other"], size=n),
        "status": np.random.choice(["Open", "Closed", "Pending"], size=n),
        "created_at": [fake.date_between(start_date="-2y", end_date="today") for _ in range(n)],
        "resolved_at": [fake.date_between(start_date="-1y", end_date="today") for _ in range(n)],
    })

# Generate
customers = generate_customer_data()
transactions = generate_transaction_data()
support = generate_support_data()

# Save CSVs
customers.to_csv("customer_data.csv", index=False)
transactions.to_csv("transactions.csv", index=False)
support.to_csv("support_tickets.csv", index=False)

print("✅ CSVs saved: customer_data.csv, transactions.csv, support_tickets.csv")

# 🔗 Simple Lineage Detection (by exact or similar names)
datasets = {
    "Customer": customers,
    "Transaction": transactions,
    "Support": support
}

matches = []
for d1_name, df1 in datasets.items():
    for d2_name, df2 in datasets.items():
        if d1_name >= d2_name:
            continue
        for c1 in df1.columns:
            for c2 in df2.columns:
                c1_clean = c1.lower().replace("_", "")
                c2_clean = c2.lower().replace("_", "")
                if c1_clean == c2_clean or c1_clean in c2_clean or c2_clean in c1_clean:
                    matches.append((d1_name, c1, d2_name, c2))

# 🧠 Visualize Lineage Network
fig = go.Figure()

# Create node layout
dataset_positions = {"Customer": 0, "Transaction": 1, "Support": 2}
node_y = 0
node_x_offset = 2

# Draw dataset nodes
for dataset, x in dataset_positions.items():
    fig.add_trace(go.Scatter(
        x=[x * node_x_offset], y=[node_y],
        mode="markers+text",
        marker=dict(size=70, color="skyblue"),
        text=[dataset],
        textposition="middle center",
        name=dataset,
        showlegend=False
    ))

# Draw connecting lines (matches)
for match in matches:
    src_dataset, src_col, tgt_dataset, tgt_col = match
    x0 = dataset_positions[src_dataset] * node_x_offset
    x1 = dataset_positions[tgt_dataset] * node_x_offset
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[0, 0],
        mode="lines+text",
        line=dict(width=1, color="gray", dash="dot"),
        text=[f"{src_col} → {tgt_col}"],
        textposition="top center",
        hoverinfo="text",
        showlegend=False
    ))

fig.update_layout(
    title="🔗 Dataset Lineage (Shared/Similar Columns)",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=400,
    margin=dict(l=50, r=50, t=60, b=40)
)

fig.show()
