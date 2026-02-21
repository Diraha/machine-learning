# IMPORT LIBRARY
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# MENYIAPKAN DATASETS
data_rumah = {
    "luas_tanah" : [90, 120, 60, 200, 150, 75, 300, 110, 50, 180],
    "luas_bangunan" : [70, 100, 45, 180, 130, 60, 250, 85, 40, 150],
    "jumlah_kamar" : [2, 3, 2, 4, 3, 2, 5, 3, 1, 4],
    "jumlah_kamar_mandi" : [1, 2, 1, 3, 2, 1, 4, 2, 1, 3],
    "harga_rumah" : [650, 1200, 400, 2500, 1600, 500, 4000, 900, 300, 2200]
}

# MENGUBAH DATASETS MENJADI DATA FRAME
data_rumah_df = pd.DataFrame(data_rumah)

# MENGUBAH DATA FRAME MENJADI NUMPY ARRAY UNTUK MEMBUAT DATA TRAINING
X_train = np.array(data_rumah_df[["luas_tanah", "luas_bangunan", "jumlah_kamar", "jumlah_kamar_mandi"]])
y_train = np.array(data_rumah["harga_rumah"])

# MEMBUAT MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# TESTING DATA
data_pred = {
    "luas_tanah" : [95, 130, 70, 250, 140],
    "luas_bangunan" : [75, 110, 55, 200, 115],
    "jumlah_kamar" : [2, 3, 2, 5, 3],
    "jumlah_kamar_mandi" : [2, 2, 1, 3, 2],
    "harga_rumah" : [800, 1400, 450, 3200, 1350]
}

data_pred_df = pd.DataFrame(data_pred)

X_test = np.array(data_pred_df[["luas_tanah", "luas_bangunan", "jumlah_kamar", "jumlah_kamar_mandi"]])
y_test = np.array(data_pred["harga_rumah"])

y_pred = model.predict(X_test)

# MELIHAT PERFORMA MODEL
performance = r2_score(y_test, y_pred)

# APLIKASI MODEL
luas_tanah = int(input("Masukkan Luas Tanah: "))
luas_bangunan = int(input("Masukkan Luas Bangunan: "))
jumlah_kamar = int(input("Masukkan Jumlah Kamar: "))
jumlah_kamar_mandi = int(input("Masukkan Jumlah Kamar Mandi: "))

result = np.array([[luas_tanah, luas_bangunan, jumlah_kamar, jumlah_kamar_mandi]])
result_pred = model.predict(result)

harga_akhir = f"Senilai: {round(result_pred[0])*1000000:,}".replace(",", ".")
print("===== HARGA RUMAH YANG ANDA INPUT =====")
print(harga_akhir)

# MEMBUAT PLOT
data_rumah_df.plot(kind="scatter", x="luas_bangunan", y="harga_rumah")

plt.title("Luas Bangunan vs Harga Rumah")
plt.xlabel("Luas Bangunan")
plt.ylabel("Harga Rumah")
plt.grid(True)
plt.show()