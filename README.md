# Ridge & Lasso Regression Project

## 1. Penjelasan Cara Kerja Ridge & Lasso

**Ridge Regression** dan **Lasso Regression** adalah teknik regularisasi pada regresi linier yang digunakan untuk mengatasi masalah multikolinearitas dan overfitting.

- **Ridge Regression** (L2 Regularization): Menambahkan penalti berupa jumlah kuadrat dari koefisien ke fungsi loss. Ridge tidak pernah benar-benar mengeliminasi fitur, hanya mengecilkan koefisien.
- **Lasso Regression** (L1 Regularization): Menambahkan penalti berupa jumlah absolut dari koefisien ke fungsi loss. Lasso dapat mengecilkan beberapa koefisien menjadi nol, sehingga dapat melakukan seleksi fitur.

## 2. Komentar di Dalam Kode

Contoh komentar sudah ditambahkan pada file `main.py` untuk setiap bagian penting, seperti:

```python
# Import library yang dibutuhkan
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
# ...existing code...

# Membaca dataset
df = pd.read_csv("boston.csv")

# Split data menjadi train dan test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Ridge Regression dengan GridSearchCV
ridge = Ridge()
param_ridge = {"alpha": [0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge, param_ridge, cv=5, scoring="r2")
ridge_cv.fit(X_train_scaled, y_train)

# Lasso Regression dengan GridSearchCV
lasso = Lasso(max_iter=10000)
param_lasso = {"alpha": [0.001, 0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso, param_lasso, cv=5, scoring="r2")
lasso_cv.fit(X_train_scaled, y_train)
```

## 3. Step by Step Membuat Ridge & Lasso

1. **Import Library**
	- Import pandas, numpy, sklearn, seaborn, matplotlib, dll.
2. **Load Dataset**
	- Baca file `boston.csv` menggunakan pandas.
3. **Eksplorasi Data**
	- Cek info, missing value, distribusi data, dan korelasi antar fitur.
4. **Preprocessing**
	- Drop kolom yang tidak diperlukan, standarisasi fitur dengan `StandardScaler`.
5. **Split Data**
	- Bagi data menjadi data latih dan data uji (`train_test_split`).
6. **Cek Multikolinearitas**
	- Hitung VIF (Variance Inflation Factor) untuk mengetahui multikolinearitas antar fitur.
7. **Ridge Regression**
	- Lakukan training model Ridge dengan `GridSearchCV` untuk mencari alpha terbaik.
	- Evaluasi model dengan R2 dan MSE.
8. **Lasso Regression**
	- Lakukan training model Lasso dengan `GridSearchCV` untuk mencari alpha terbaik.
	- Evaluasi model dengan R2 dan MSE.
9. **Interpretasi Hasil**
	- Bandingkan performa Ridge dan Lasso, serta perhatikan fitur mana yang dieliminasi oleh Lasso.

---

Untuk detail kode dan komentar, silakan lihat file `main.py`.