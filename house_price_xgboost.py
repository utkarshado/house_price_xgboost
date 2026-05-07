import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# XGBoost
from xgboost import XGBRegressor

r = 74723348

d = pd.read_csv(
    r"C:\Users\utkar\Downloads\proj\practice_independency\house-prices-advanced-regression-techniques\train.csv"
)

# preprocessing :

# correlation :
corr = d.corr(numeric_only=True)

# removing columns with more than 40% missing values
missing = d.isnull().mean() * 100

drop_col = missing[missing > 40].index
d = d.drop(columns=drop_col)

# removing outliers :
d = d.drop(d[(d['GrLivArea'] > 4000) & (d['SalePrice'] < 300000)].index)

# defining target/label and features/predictors
y = np.log1p(d['SalePrice'])
x = d.drop("SalePrice", axis=1)

# feature engineering :
x['TotalSF'] = (x['TotalBsmtSF'] + x['1stFlrSF'] + x['2ndFlrSF'])

x['TotalBath'] = (x['FullBath'] + 0.5 * x['HalfBath'] + x['BsmtFullBath'] + 0.5 * x['BsmtHalfBath'])

x['HouseAge'] = x['YrSold'] - x['YearBuilt']

x['RemodelAge'] = x['YrSold'] - x['YearRemodAdd']

# missing value fixing : both numerical and categorical
num_col = x.select_dtypes(include=["int64", "float64"]).columns
cat_col = x.select_dtypes(include=["object", "string"]).columns

x[num_col] = x[num_col].fillna(x[num_col].median())
x[cat_col] = x[cat_col].fillna("Missing")

# one hot encoding
x = pd.get_dummies(x)


# visualisations :::

# SalePrice distribution
plt.figure(figsize=(8, 5))

sns.histplot(data=d, x="SalePrice", bins=50, kde=True)

plt.title("SalePrice Distribution")
plt.savefig("saleprice_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# Correlation heatmap (top correlated features)
top_corr = corr["SalePrice"].abs().sort_values(ascending=False).head(10).index

plt.figure(figsize=(10, 7))

sns.heatmap(d[top_corr].corr(),annot=True,cmap="coolwarm",fmt=".2f")

plt.title("Top Features Correlated With SalePrice")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# GrLivArea vs SalePrice
plt.figure(figsize=(8, 5))

sns.scatterplot(data=d,x="GrLivArea",y="SalePrice")

plt.title("GrLivArea vs SalePrice")
plt.tight_layout()
plt.savefig("grlivarea_vs_saleprice.png", dpi=300, bbox_inches="tight")
plt.show()


# OverallQual vs SalePrice
plt.figure(figsize=(8, 5))

sns.boxplot(data=d,x="OverallQual",y="SalePrice")

plt.title("OverallQual vs SalePrice")
plt.tight_layout()
plt.savefig("overallqual_vs_saleprice.png", dpi=300, bbox_inches="tight")
plt.show()

# training
X_train, X_val, Y_train, Y_val = train_test_split(
    x, y, test_size=0.2, random_state=r
)

# xgboost

mod = XGBRegressor(
    objective='reg:squarederror',
    random_state=r
)

# hyperparameter tuning
param_grid = {
    'n_estimators': [200, 300, 500, 800],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.01, 0.1],
    'reg_alpha': [0, 0.001, 0.01],
    'reg_lambda': [1, 1.5, 2]

}

ran_search = RandomizedSearchCV(
    estimator=mod,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    verbose=2,
    random_state=r,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error"
)

# training
ran_search.fit(X_train, Y_train)

# best model
model = ran_search.best_estimator_

# prediction
pred = np.expm1(model.predict(X_val))
actual = np.expm1(Y_val)

# evaluation :
rmse = np.sqrt(mean_squared_error(actual, pred))


# Actual vs Predicted
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=actual,
    y=pred
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()


# Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


print("Best Parameters : ", ran_search.best_params_)
print("The root mean squared error is : ", rmse)