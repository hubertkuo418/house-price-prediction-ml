import pandas as pd
import joblib
import numpy as np

from evaluate import evaluate 

# =========================
# 1. Load model
# =========================
model = joblib.load("models/house_price_model.pkl")

# =========================
# 2. Load test data
# =========================
df = pd.read_csv("data/train.csv") 

y_true = df["SalePrice"]

features = [
    'OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',
    '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd',
    'MSZoning','Neighborhood','HouseStyle','Exterior1st','KitchenQual'
]

X = df[features]

# =========================
# 3. Prediction
# =========================
y_pred = model.predict(X)

# =========================
# 4. Show sample results
# =========================
print("\n===== Sample Predictions =====")
for i in range(5):
    print(f"True: {y_true.iloc[i]:.0f} | Pred: {y_pred[i]:.0f}")

# =========================
# 5. Evaluation
# =========================
print("\n===== Model Evaluation =====")
evaluate(y_true, y_pred)