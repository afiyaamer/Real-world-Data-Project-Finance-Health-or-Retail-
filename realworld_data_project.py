# ============================================================
# Real-World Data Project — Retail Sales Analysis & Prediction
# ============================================================
# Domain  : Retail
# Dataset : Synthetic Superstore-style Sales Data
# Goal    : End-to-end EDA + Sales Prediction + Visualisations
# Tools   : Python, Pandas, Scikit-learn, Matplotlib, Seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# ── 1. Generate Realistic Retail Dataset ─────────────────────
n = 1000
categories    = ["Electronics", "Clothing", "Furniture", "Office Supplies", "Food & Beverages"]
regions       = ["North", "South", "East", "West"]
payment_modes = ["Credit Card", "Cash", "UPI", "Debit Card"]

df = pd.DataFrame({
    "Order_ID":      [f"ORD-{1000+i}" for i in range(n)],
    "Date":          pd.date_range(start="2023-01-01", periods=n, freq="8h"),
    "Category":      np.random.choice(categories, n, p=[0.25, 0.20, 0.15, 0.25, 0.15]),
    "Region":        np.random.choice(regions, n),
    "Payment_Mode":  np.random.choice(payment_modes, n),
    "Quantity":      np.random.randint(1, 15, n),
    "Unit_Price":    np.round(np.random.uniform(10, 500, n), 2),
    "Discount":      np.round(np.random.uniform(0, 0.40, n), 2),
    "Shipping_Cost": np.round(np.random.uniform(5, 50, n), 2),
})

df["Sales"]  = np.round(df["Quantity"] * df["Unit_Price"] * (1 - df["Discount"]), 2)
df["Profit"] = np.round(df["Sales"] * np.random.uniform(0.05, 0.35, n) - df["Shipping_Cost"], 2)
df["Month"]  = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")

print("=" * 55)
print("RETAIL SALES DATASET — OVERVIEW")
print("=" * 55)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df[['Sales', 'Profit', 'Quantity', 'Discount']].describe().round(2)}")

# ── 2. Key Business Insights ──────────────────────────────────
print("\n" + "=" * 55)
print("KEY BUSINESS INSIGHTS")
print("=" * 55)
print(f"Total Revenue  : ₹{df['Sales'].sum():,.2f}")
print(f"Total Profit   : ₹{df['Profit'].sum():,.2f}")
print(f"Avg Order Value: ₹{df['Sales'].mean():,.2f}")
print(f"Profit Margin  : {(df['Profit'].sum()/df['Sales'].sum())*100:.2f}%")
print(f"\nSales by Category:\n{df.groupby('Category')['Sales'].sum().sort_values(ascending=False).round(2)}")
print(f"\nSales by Region:\n{df.groupby('Region')['Sales'].sum().sort_values(ascending=False).round(2)}")

# ── 3. Machine Learning — Sales Prediction ───────────────────
le = LabelEncoder()
df_ml = df.copy()
for col in ["Category", "Region", "Payment_Mode"]:
    df_ml[col] = le.fit_transform(df_ml[col])

features = ["Category", "Region", "Quantity", "Unit_Price", "Discount", "Shipping_Cost", "Month"]
X = df_ml[features]
y = df_ml["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Linear Regression":    LinearRegression(),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "=" * 55)
print("MODEL PERFORMANCE — SALES PREDICTION")
print("=" * 55)
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "mae": mae, "rmse": rmse, "r2": r2}
    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f}  |  RMSE: {rmse:.2f}  |  R²: {r2:.4f}")

# ── 4. Visualisations ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("Real-World Retail Data Project — Sales Analysis & Prediction",
             fontsize=16, fontweight="bold", y=0.98)

colors_cat = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"]

# Plot 1: Sales by Category
ax1 = fig.add_subplot(3, 3, 1)
cat_sales = df.groupby("Category")["Sales"].sum().sort_values()
cat_sales.plot(kind="barh", ax=ax1, color=colors_cat, edgecolor="white")
ax1.set_title("Total Sales by Category", fontweight="bold")
ax1.set_xlabel("Sales (₹)")
ax1.grid(axis="x", alpha=0.3)

# Plot 2: Sales by Region (Pie)
ax2 = fig.add_subplot(3, 3, 2)
reg_sales = df.groupby("Region")["Sales"].sum()
ax2.pie(reg_sales, labels=reg_sales.index, autopct="%1.1f%%",
        colors=["#4C72B0","#55A868","#C44E52","#8172B2"], startangle=140)
ax2.set_title("Sales Distribution by Region", fontweight="bold")

# Plot 3: Monthly Sales Trend
ax3 = fig.add_subplot(3, 3, 3)
monthly = df.groupby("Month")["Sales"].sum()
ax3.plot(monthly.index, monthly.values, marker="o", color="#4C72B0", linewidth=2.5)
ax3.fill_between(monthly.index, monthly.values, alpha=0.15, color="#4C72B0")
ax3.set_title("Monthly Sales Trend", fontweight="bold")
ax3.set_xlabel("Month")
ax3.set_ylabel("Sales (₹)")
ax3.set_xticks(range(1, 13))
ax3.grid(alpha=0.3)

# Plot 4: Profit by Category (Box Plot)
ax4 = fig.add_subplot(3, 3, 4)
df.boxplot(column="Profit", by="Category", ax=ax4,
           patch_artist=True, notch=False)
ax4.set_title("Profit Distribution by Category", fontweight="bold")
ax4.set_xlabel("Category")
ax4.set_ylabel("Profit (₹)")
plt.sca(ax4)
plt.xticks(rotation=20, ha="right", fontsize=8)
fig.suptitle("Real-World Retail Data Project — Sales Analysis & Prediction",
             fontsize=16, fontweight="bold")

# Plot 5: Discount vs Sales Scatter
ax5 = fig.add_subplot(3, 3, 5)
for i, cat in enumerate(categories):
    sub = df[df["Category"] == cat]
    ax5.scatter(sub["Discount"], sub["Sales"], alpha=0.4, s=20,
                color=colors_cat[i], label=cat)
ax5.set_title("Discount vs Sales", fontweight="bold")
ax5.set_xlabel("Discount")
ax5.set_ylabel("Sales (₹)")
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

# Plot 6: Payment Mode Distribution
ax6 = fig.add_subplot(3, 3, 6)
pay_counts = df["Payment_Mode"].value_counts()
ax6.bar(pay_counts.index, pay_counts.values,
        color=["#4C72B0","#55A868","#C44E52","#8172B2"], edgecolor="white")
ax6.set_title("Orders by Payment Mode", fontweight="bold")
ax6.set_ylabel("Number of Orders")
ax6.grid(axis="y", alpha=0.3)

# Plot 7: Actual vs Predicted (Best Model — Random Forest)
ax7 = fig.add_subplot(3, 3, 7)
rf_preds = results["Random Forest"]["preds"]
ax7.scatter(y_test, rf_preds, alpha=0.4, color="#4C72B0", s=20)
mn, mx = y_test.min(), y_test.max()
ax7.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect Fit")
ax7.set_title("Actual vs Predicted Sales\n(Random Forest)", fontweight="bold")
ax7.set_xlabel("Actual Sales (₹)")
ax7.set_ylabel("Predicted Sales (₹)")
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Model R² Comparison
ax8 = fig.add_subplot(3, 3, 8)
model_names = list(results.keys())
r2_scores   = [results[m]["r2"] for m in model_names]
bars = ax8.bar(model_names, r2_scores, color=["#4C72B0","#55A868","#C44E52"], edgecolor="white")
ax8.set_ylim([0, 1.05])
ax8.set_title("Model R² Score Comparison", fontweight="bold")
ax8.set_ylabel("R² Score")
ax8.set_xticks(range(len(model_names)))
ax8.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
for bar, r2 in zip(bars, r2_scores):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{r2:.3f}", ha="center", fontsize=9, fontweight="bold")
ax8.grid(axis="y", alpha=0.3)

# Plot 9: Feature Importance (Random Forest)
ax9 = fig.add_subplot(3, 3, 9)
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=features).sort_values()
fi.plot(kind="barh", ax=ax9, color="#4C72B0", edgecolor="white")
ax9.set_title("Feature Importance\n(Random Forest)", fontweight="bold")
ax9.set_xlabel("Importance Score")
ax9.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("retail_analysis_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as retail_analysis_results.png")

# ── 5. Conclusions ────────────────────────────────────────────
print("\n" + "=" * 55)
print("CONCLUSIONS")
print("=" * 55)
best = max(results, key=lambda m: results[m]["r2"])
print(f"Best Model     : {best} (R² = {results[best]['r2']:.4f})")
print(f"Top Category   : {df.groupby('Category')['Sales'].sum().idxmax()}")
print(f"Top Region     : {df.groupby('Region')['Sales'].sum().idxmax()}")
print(f"Peak Month     : {df.groupby('Month')['Sales'].sum().idxmax()}")
print(f"Avg Discount   : {df['Discount'].mean()*100:.1f}%")
print("=" * 55)
