import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

# Load dataset
dataset = pd.read_excel("data\hospital_equipment_data.xlsx")

# Target & features
y = dataset['Remaining_Useful_Life_Years']
X = dataset.drop('Remaining_Useful_Life_Years', axis=1)

# Handle categorical variables
X = pd.get_dummies(X, drop_first=False)

# Important features for final model
important_features = [
    'Equipment_Age_Years',
    'Error_Count',
    'Downtime_Duration',
    'Maintenance_Count'
] + [col for col in X.columns if col.startswith("Equipment_Type_")]

X_final = X[important_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ========== Model Training ==========
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Gradient Boosting RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Save model & scaler
joblib.dump(model, "gbr_model.pkl")
print("Model saved successfully!")