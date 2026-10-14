from pathlib import Path
import pandas as pd

folder = Path("data")
files = [
    "ravenstack_accounts.csv",
    "ravenstack_churn_events.csv",
    "ravenstack_feature_usage.csv",
    "ravenstack_subscriptions.csv",
    "ravenstack_support_tickets.csv",
]

for name in files:
    df = pd.read_csv(folder / name)
    print("FILE:", name)
    print("Shape:", df.shape)
    print("\nColumns:",df.columns.tolist())
    print("\nFirst 3 rows:",df.head(3).to_string(index=False))
    print("\nMissing values:",df.isnull().sum().to_string())
