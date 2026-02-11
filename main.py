import pandas as pd  # Library untuk data processing
from sklearn.linear_model import Ridge, Lasso  # Import model Ridge & Lasso
import seaborn as sns  # Visualisasi data
import matplotlib.pyplot as plt  # Visualisasi data
from sklearn.preprocessing import StandardScaler  # Standarisasi fitur

# 1. Load dataset
df = pd.read_csv("boston.csv")

# 2. Cek duplikasi pada setiap kolom
print(f"crim = {df.crim.duplicated().sum()}")
print(f"zn = {df.zn.duplicated().sum()}")
print(f"indus = {df.indus.duplicated().sum()}")
print(f"chas = {df.chas.duplicated().sum()}")
print(f"nox = {df.nox.duplicated().sum()}")
print(f"rm = {df.rm.duplicated().sum()}")
print(f"age = {df.age.duplicated().sum()}")
print(f"dis = {df.dis.duplicated().sum()}")
print(f"rad = {df.rad.duplicated().sum()}")
print(f"tax = {df.tax.duplicated().sum()}")
print(f"ptratio = {df.ptratio.duplicated().sum()}")
print(f"black = {df.black.duplicated().sum()}")
print(f"lstat = {df.lstat.duplicated().sum()}")
print(f"medv = {df.medv.duplicated().sum()}")

# 3. Info dataset
df.info()

# 4. Cek missing value
df.isna().sum()

# 5. Sampling dan value count
df.zn.sample(10)
df.chas.sample(10)
df.age.value_counts()

# 6. Visualisasi distribusi dan count
sns.countplot(y = df.zn)
sns.histplot(df["medv"], kde=True)
plt.title("Distribution of MEDV")
plt.show()

sns.scatterplot(
    x="tax",
    y="medv",
    hue="medv",
    palette="viridis",
    data=df
  axes[1].set_title('Box Plot')

  data.plot.hist(ax=axes[2])
  axes[2].set_title('Histogram')

  plt.tight_layout()
  plt.show()
  
  
df_process = df.copy()
  
  
corr = df.corr(numeric_only=True)

plt.figure(figsize=(14,10))
    annot=True,
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score


plt.show()

# 7. Scatterplot hubungan tax dan medv
sns.scatterplot(
    x="tax",
    y="medv",
    hue="medv",
    palette="viridis",
    data=df
)

import statsmodels.api as sm
# Fungsi untuk menampilkan distribusi data
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = X.copy()

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]


# 8. Copy data untuk preprocessing
vif_data.sort_values("VIF", ascending=False)

# 9. Korelasi antar fitur
corr = df.corr(numeric_only=True)
plt.figure(figsize=(14,10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Heatmap Korelasi Semua Kolom")
plt.show()

# 10. Drop kolom yang tidak diperlukan



# 11. Pisahkan fitur dan target
X = df.drop(columns=["medv"])
ridge = Ridge()

# 12. Split data menjadi train dan test
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_ridge = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

ridge_cv = GridSearchCV(
    ridge,
    param_ridge,
    cv=5,
    scoring="r2"
)

ridge_cv.fit(X_train_scaled, y_train)

best_ridge = ridge_cv.best_estimator_

y_test_pred_ridge = best_ridge.predict(X_test_scaled)

print("Ridge Regression")
print("Best alpha:", ridge_cv.best_params_)
print("R2 Test:", r2_score(y_test, y_test_pred_ridge))
print("MSE Test:", mean_squared_error(y_test, y_test_pred_ridge))

# 13. Standarisasi fitur agar model lebih stabil
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# 14. Cek multikolinearitas dengan VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = X.copy()
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]
vif_data.sort_values("VIF", ascending=False)

# 15. Ridge Regression
ridge = Ridge()
param_ridge = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}
ridge_cv = GridSearchCV(
    ridge,
    param_ridge,
    cv=5,
    scoring="r2"
)
ridge_cv.fit(X_train_scaled, y_train)
best_ridge = ridge_cv.best_estimator_
y_test_pred_ridge = best_ridge.predict(X_test_scaled)
print("Ridge Regression")
print("Best alpha:", ridge_cv.best_params_)
print("R2 Test:", r2_score(y_test, y_test_pred_ridge))
print("MSE Test:", mean_squared_error(y_test, y_test_pred_ridge))

# 16. Lasso Regression
lasso = Lasso(max_iter=10000)
param_lasso = {
    "alpha": [0.001, 0.01, 0.1, 1, 10]
}
lasso_cv = GridSearchCV(
    lasso,
    param_lasso,
    cv=5,
    scoring="r2"
)
lasso_cv.fit(X_train_scaled, y_train)
best_lasso = lasso_cv.best_estimator_
y_test_pred_lasso = best_lasso.predict(X_test_scaled)
print("Lasso Regression")
print("Best alpha:", lasso_cv.best_params_)
print("R2 Test:", r2_score(y_test, y_test_pred_lasso))
print("MSE Test:", mean_squared_error(y_test, y_test_pred_lasso))
