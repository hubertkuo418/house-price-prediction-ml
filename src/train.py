import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# =====================
# Load data
# =====================
train = pd.read_csv("data/train.csv")

numerical_features = [
    'OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',
    '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd'
]

categorical_features = [
    'MSZoning','Neighborhood','HouseStyle','Exterior1st','KitchenQual'
]

X = train[numerical_features + categorical_features]
y = train["SalePrice"]

# =====================
# preprocessing
# =====================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, numerical_features),
    ("cat", cat_pipe, categorical_features)
])

# =====================
# model
# =====================
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model.fit(X, y)

joblib.dump(model, "models/house_price_model.pkl")

print("Training completed & model saved.")