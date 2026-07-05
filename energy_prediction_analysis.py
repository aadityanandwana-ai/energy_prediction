"""
Appliances Energy Prediction - Full Analysis Pipeline
Dataset: energydata_complete.csv (UCI Appliances Energy Prediction Data Set)
Target: Appliances (Wh)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("energydata_complete.csv")
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
print("Shape:", df.shape)
print(df.info())
print(df.describe().T)

# ----------------------------------------------------------------------
# 2. BASIC EDA
# ----------------------------------------------------------------------
print("\nMissing values:\n", df.isnull().sum().sum())

plt.figure(figsize=(10, 4))
sns.histplot(df["Appliances"], bins=50, kde=True)
plt.title("Distribution of Appliances Energy Use (Wh)")
plt.xlabel("Wh")
plt.tight_layout()
plt.savefig("appliances_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["Appliances"], lw=0.4)
plt.title("Appliances Energy Use Over Time")
plt.xlabel("Date")
plt.ylabel("Wh")
plt.tight_layout()
plt.savefig("appliances_timeseries.png", dpi=150)
plt.close()

# Correlation heatmap (numeric only, exclude date)
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# Top correlations with target
top_corr = corr["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False)
print("\nTop correlated features with Appliances:\n", top_corr.head(10))

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["month"] = df["date"].dt.month

# Drop the two random noise variables (rv1, rv2) - known to be non-informative
drop_cols = ["date", "rv1", "rv2"]
features = [c for c in df.columns if c not in drop_cols + ["Appliances"]]

X = df[features]
y = df["Appliances"]

# ----------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT + SCALING
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 5. MODEL TRAINING & COMPARISON
# ----------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = []
for name, model in models.items():
    if name in ["Random Forest", "Gradient Boosting"]:
        model.fit(X_train, y_train)          # tree models don't need scaling
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append([name, rmse, mae, r2])
    print(f"{name:20s} RMSE={rmse:8.2f}  MAE={mae:8.2f}  R2={r2:.4f}")

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n", results_df)

# ----------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# ----------------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
importances.head(15).plot(kind="barh")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nTop 15 important features:\n", importances.head(15))

# ----------------------------------------------------------------------
# 7. SAVE BEST MODEL'S PREDICTIONS PLOT
# ----------------------------------------------------------------------
best_model_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
print(f"\nBest model: {best_model_name}")

best_model = models[best_model_name]
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    preds_best = best_model.predict(X_test)
else:
    preds_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds_best, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Appliances (Wh)")
plt.ylabel("Predicted Appliances (Wh)")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

print("\nAll plots and results saved to current directory.")
