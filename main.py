import pandas as pd
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("boston.csv")
df

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


df.info(
)

df.isna().sum()

df.zn.sample(10)

df.chas.sample(10)

df.age.value_counts()

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
)


import statsmodels.api as sm
def show_distributin(data:pd.Series):
  fig,axes = plt.subplots(nrows=1,ncols=3)
  fig.set_size_inches(12,4)

  sm.qqplot(data, line='r', ax = axes[0])
  axes[0].set_title('Q-Q plot')

  data.plot.box(ax=axes[1])
  axes[1].set_title('Box Plot')

  data.plot.hist(ax=axes[2])
  axes[2].set_title('Histogram')

  plt.tight_layout()
  plt.show()
  
  
df_process = df.copy()
  
  
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


df_process.drop(columns=["rad",'dis'], inplace=True)
df_process.info()


X = df.drop(columns=["medv"])
y = df["medv"]


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = X.copy()

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif_data.sort_values("VIF", ascending=False)


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
