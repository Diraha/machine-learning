# MENGAMBIL LIBRARY YANG DIBUTUHKAN
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import joblib

# MEMBUAT DATA TRAINING
data_training = {
    "height" : [158, 162, 150, 170, 180, 155, 168, 160, 178, 182, 170, 188, 176, 185, 172, 175],
    "weight" : [52, 55, 48, 65, 80, 50, 60, 54, 75, 85, 68, 92, 74, 90, 70, 72],
    "gender" : ["F", "F", "F", "F", "M", "F", "F", "F", "M", "M", "M", "M", "M", "M", "M", "F"]
}

# MENGUBAH DATA TRAINING MENJADI DATAFRAME
df_training = pd.DataFrame(data_training)

# PRE-PROCESSING DATA TARGET
y_train = df_training["gender"].values.reshape(-1, 1)
onehot_encoder_training = OneHotEncoder()
y_train = onehot_encoder_training.fit_transform(y_train).toarray()

# MENGUBAH BENTUK DATA TRAINING MENJADI MATRIKS
X_train = np.array(df_training[["height", "weight"]])

# MEMBUAT MODEL DAN MELAKUKAN TRAINING MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# MEMBUAT DATA TESTING
data_testing = {
    "height" : [165, 152, 172, 159, 177, 183, 169, 190],
    "weight" : [58, 46, 68, 53, 78, 88, 66, 95],
    "gender" : ["F", "F", "F", "F", "M", "M", "M", "M"]
}

# MENGUBAH DATA TESTING MENJADI DATA FRAME
df_testing = pd.DataFrame(data_testing)

# PRE-PROCESSING DATA TARGET
y_test = df_testing["gender"].values.reshape(-1, 1)
onehot_encoder_testing = OneHotEncoder()
y_test = onehot_encoder_testing.fit_transform(y_test).toarray()

# MENGUBAH DATA TESTING MENJADI MATRIKS
X_test = np.array(df_testing[["height", "weight"]])

# MELAKUKAN TESTING MODEL
preds = model.predict(X_test)

# MENGUBAH BENTUK HASIL PREDIKSI & TESTING DATA MENJADI NILAI DISKRIT (KELAS)
preds_label = np.argmax(preds, axis=1)
y_test_label = np.argmax(y_test, axis=1)

# MENGECEK AKURASI SCORE
acc = accuracy_score(y_test_label, preds_label)

# UJICOBA MODEL ML
labels = onehot_encoder_testing.categories_[0]
height = int(input("Masukkan Tinggi Badan Anda: "))
weight = int(input("Masukkan Berat Badan Anda: "))

data_ujicoba = np.array([[height, weight]])

preds_ujicoba = model.predict(data_ujicoba)
preds_ujicoba_label = np.argmax(preds_ujicoba)
print(labels[preds_ujicoba_label])

# SAVING MODEL
# joblib.dump(model, "MODEL/gender_classifier_lr.joblib")