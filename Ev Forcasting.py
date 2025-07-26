import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1️⃣ Load and clean the data
df = pd.read_csv("3ae033f50fa345051652.csv")  # Update with your actual file name
df['Electric Vehicle (EV) Total'] = df['Electric Vehicle (EV) Total'].str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df[(df['State'] == 'WA') & df['County'].notna() & df['Date'].notna()].copy()

# 2️⃣ Sort and create time-based features
df = df.sort_values(["County", "Date"])
df["months_since_start"] = df.groupby("County")["Date"].transform(lambda x: (x - x.min()).dt.days // 30)

# 3️⃣ Lag & rolling features
df["ev_total_lag1"] = df.groupby("County")["Electric Vehicle (EV) Total"].shift(1)
df["ev_total_lag2"] = df.groupby("County")["Electric Vehicle (EV) Total"].shift(2)
df["ev_total_lag3"] = df.groupby("County")["Electric Vehicle (EV) Total"].shift(3)
df["ev_total_roll_mean_3"] = df[["ev_total_lag1", "ev_total_lag2", "ev_total_lag3"]].mean(axis=1)
df["ev_total_pct_change_1"] = df.groupby("County")["Electric Vehicle (EV) Total"].pct_change(periods=1)
df["ev_total_pct_change_3"] = df.groupby("County")["Electric Vehicle (EV) Total"].pct_change(periods=3)
df["cumulative_ev"] = df.groupby("County")["Electric Vehicle (EV) Total"].cumsum()

# 4️⃣ Growth slope feature
def compute_growth_slope(group):
    if len(group) < 2:
        return pd.Series([0]*len(group), index=group.index)
    X = group["months_since_start"].values.reshape(-1, 1)
    y = group["Electric Vehicle (EV) Total"].values
    model = LinearRegression().fit(X, y)
    return pd.Series(model.predict(X), index=group.index)

df["ev_growth_slope"] = df.groupby("County", group_keys=False).apply(compute_growth_slope)

# 5️⃣ Drop incomplete rows and handle infinities
df = df.dropna(subset=[
    "ev_total_lag1", "ev_total_lag2", "ev_total_lag3",
    "ev_total_roll_mean_3", "ev_total_pct_change_1",
    "ev_total_pct_change_3", "cumulative_ev", "ev_growth_slope"
])
df.replace([np.inf, -np.inf], 0, inplace=True)

# 6️⃣ Train the model
features = [
    'months_since_start', 'ev_total_lag1', 'ev_total_lag2', 'ev_total_lag3',
    'ev_total_roll_mean_3', 'ev_total_pct_change_1', 'ev_total_pct_change_3',
    'cumulative_ev', 'ev_growth_slope'
]
target = 'Electric Vehicle (EV) Total'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Forecast for next 6 months
latest_per_county = df.sort_values("Date").groupby("County").tail(1)
forecast_rows = []

for month in range(1, 7):
    for idx, row in latest_per_county.iterrows():
        county = row['County']
        future = row.copy()
        future_date = pd.to_datetime(future['Date']) + pd.DateOffset(months=1)
        future['months_since_start'] += 1

        lag1 = future['Electric Vehicle (EV) Total']
        lag2 = future['ev_total_lag1']
        lag3 = future['ev_total_lag2']
        roll_mean_3 = np.mean([lag1, lag2, lag3])
        pct1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0

        row_data = {
            'months_since_start': future['months_since_start'],
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean_3,
            'ev_total_pct_change_1': pct1,
            'ev_total_pct_change_3': pct3,
            'cumulative_ev': future['cumulative_ev'] + lag1,
            'ev_growth_slope': future['ev_growth_slope']
        }

        prediction = model.predict(pd.DataFrame([row_data]).replace([np.inf, -np.inf], 0))[0]
        forecast_rows.append({
            'County': county,
            'Date': future_date.strftime('%Y-%m-%d'),
            'Predicted EVs': round(prediction, 2)
        })

        # Update for next round
        future['Date'] = future_date
        future['ev_total_lag3'] = lag2
        future['ev_total_lag2'] = lag1
        future['ev_total_lag1'] = prediction
        future['Electric Vehicle (EV) Total'] = prediction
        future['cumulative_ev'] += prediction
        latest_per_county.loc[idx] = future

# 8️⃣ Save Forecasts
forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv("wa_county_ev_forecast.csv", index=False)
print("✅ Forecast saved: wa_county_ev_forecast.csv")

with pd.ExcelWriter("wa_county_ev_forecast.xlsx", engine="xlsxwriter") as writer:
    for county in forecast_df['County'].unique():
        df_county = forecast_df[forecast_df['County'] == county]
        df_county.to_excel(writer, sheet_name=county[:31], index=False)
print("📁 Excel saved: wa_county_ev_forecast.xlsx")

# 9️⃣ Summary Table
summary = forecast_df.groupby("County")["Predicted EVs"].sum().reset_index()
summary.columns = ["County", "6-Month Predicted Total"]
top5 = summary.sort_values("6-Month Predicted Total", ascending=False).head(5)
bottom5 = summary.sort_values("6-Month Predicted Total").head(5)

print("\n📈 Top 5 Counties by EV Growth:\n", top5)
print("\n📉 Bottom 5 Counties by EV Growth:\n", bottom5)

# 🔟 Plot one county (optional)
import matplotlib.pyplot as plt

county_to_plot = "King"  # Change to any valid WA county name
county_data = forecast_df[forecast_df["County"] == county_to_plot]
if not county_data.empty:
    plt.figure(figsize=(8, 4))
    plt.plot(county_data['Date'], county_data['Predicted EVs'], marker='o')
    plt.title(f"EV Forecast – {county_to_plot} County (Next 6 Months)")
    plt.xlabel("Month")
    plt.ylabel("Predicted EVs")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️ No data available for county: {county_to_plot}")
