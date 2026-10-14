from pathlib import Path
import pandas as pd


folder = Path("data")
accounts = pd.read_csv(folder / "ravenstack_accounts.csv")
churn = pd.read_csv(folder / "ravenstack_churn_events.csv")
usage = pd.read_csv(folder / "ravenstack_feature_usage.csv")
subs = pd.read_csv(folder / "ravenstack_subscriptions.csv")
tickets = pd.read_csv(folder / "ravenstack_support_tickets.csv")


account_ids = set(accounts["account_id"])
for name, df in [
    ("Churn events", churn),
    ("Subscriptions", subs),
    ("Support tickets", tickets),
]:
    print(
        f"{name} with unknown accounts:",
        (~df["account_id"].isin(account_ids)).sum(),
    )

subscription_ids = set(subs["subscription_id"])
print(
    "Usage with unknown subscriptions:",
    (~usage["subscription_id"].isin(subscription_ids)).sum(),
)


for col in ["signup_date"]:
    accounts[col] = pd.to_datetime(accounts[col], errors="coerce")

for col in ["start_date", "end_date"]:
    subs[col] = pd.to_datetime(subs[col], errors="coerce")

churn["churn_date"] = pd.to_datetime(churn["churn_date"], errors="coerce")

print("\nAccount churn flags:")
print(accounts["churn_flag"].value_counts(dropna=False))

print("\nSubscription churn flags:")
print(subs["churn_flag"].value_counts(dropna=False))

print("\nChurn reasons:")
print(churn["reason_code"].value_counts(dropna=False))

print("\nDate ranges:")
print("Signup:", accounts["signup_date"].min(), "to", accounts["signup_date"].max())
print("Subscription:", subs["start_date"].min(), "to", subs["start_date"].max())
print("Churn:", churn["churn_date"].min(), "to", churn["churn_date"].max())

print("\nInvalid subscription dates:")
print("End before start:", (subs["end_date"] < subs["start_date"]).sum())

print("\nChurn events per account:")
print(churn.groupby("account_id").size().describe())

print("\n--- CUSTOMER CHURN ANALYSIS ---")
accounts["churn_flag"] = accounts["churn_flag"].astype(bool)

print("\nOverall account churn rate:")
print(f"{accounts['churn_flag'].mean() * 100:.2f}%")

print("\nChurn by industry:")
industry_churn = accounts.groupby("industry")["churn_flag"].agg(
    customers="count",
    churned="sum",
    churn_rate="mean",
)
industry_churn["churn_rate"] = (industry_churn["churn_rate"] * 100).round(2)
print(industry_churn)

print("\nChurn by plan:")
plan_churn = accounts.groupby("plan_tier")["churn_flag"].agg(
    customers="count",
    churned="sum",
    churn_rate="mean",
)
plan_churn["churn_rate"] = (plan_churn["churn_rate"] * 100).round(2)
print(plan_churn)

print("\nChurn by country:")
country_churn = accounts.groupby("country")["churn_flag"].agg(
    customers="count",
    churned="sum",
    churn_rate="mean",
)
country_churn["churn_rate"] = (country_churn["churn_rate"] * 100).round(2)
print(country_churn)



usage_data = usage.merge(
    subs[["subscription_id", "account_id"]],
    on="subscription_id",
    how="left"
)

usage_data = usage_data.merge(
    accounts[["account_id", "churn_flag"]],
    on="account_id",
    how="left"
)

print("\nAverage usage by account churn status:")
print(
    usage_data.groupby("churn_flag").agg(
        usage_events=("usage_id", "count"),
        avg_usage_count=("usage_count", "mean"),
        avg_errors=("error_count", "mean"),
        avg_duration=("usage_duration_secs", "mean")
    ).round(2)
)


support_data = tickets.merge(
    accounts[["account_id", "churn_flag"]],
    on="account_id",
    how="left"
)

print("\nSupport experience by account churn status:")
print(
    support_data.groupby("churn_flag").agg(
        ticket_count=("ticket_id", "count"),
        avg_resolution_hours=("resolution_time_hours", "mean"),
        avg_first_response_minutes=("first_response_time_minutes", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        escalation_rate=("escalation_flag", "mean")
    ).round(2)
)




subs["start_date"] = pd.to_datetime(
    subs["start_date"], errors="coerce"
)
subs["end_date"] = pd.to_datetime(
    subs["end_date"], errors="coerce"
)


observation_date = max(
    accounts["signup_date"].max(),
    subs["start_date"].max(),
    churn["churn_date"].max()
)

subs["lifetime_days"] = (
    subs["end_date"].fillna(observation_date)
    - subs["start_date"]
).dt.days

print("Observation cutoff:", observation_date)

print("\nAverage subscription duration by churn status:")
print(
    subs.groupby("churn_flag")["lifetime_days"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(1)
)

print("\nSubscription lifetime by plan:")
print(
    subs.groupby("plan_tier")["lifetime_days"]
    .agg(["count", "mean", "median"])
    .round(1)
)
