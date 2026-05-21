import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -----------------------------
# CARGAR DATASET
# -----------------------------

df = pd.read_csv("dataset.csv")

print(df.head())

# -----------------------------
# SEPARAR DATOS
# -----------------------------

X = df.drop("label", axis=1).values

y = df["label"].values

# -----------------------------
# CONVERTIR LETRAS A NUMEROS
# -----------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# Ejemplo:
# A=0 B=1 C=2

# Convertir a categórico
y = to_categorical(y)

# -----------------------------
# DIVIDIR DATOS
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# CREAR RED NEURONAL
# -----------------------------

model = Sequential()

# Entrada
model.add(Dense(256, activation='relu', input_shape=(126,)))

# Capas ocultas
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))

# Salida automática
model.add(Dense(y.shape[1], activation='softmax'))

# -----------------------------
# COMPILAR
# -----------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# ENTRENAR IA
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# EVALUAR
# -----------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Precision final: {accuracy * 100:.2f}%")

# -----------------------------
# GUARDAR MODELO
# -----------------------------

model.save("modelo_senas.h5")

# Guardar etiquetas
np.save("labels.npy", encoder.classes_)

print("Modelo guardado correctamente")