import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

from collections import deque
# Cambiamos load_model por la estructura secuencial segura
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# -----------------------------
# VOZ
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -----------------------------
# IA (Reconstrucción explícita y segura)
# -----------------------------
labels = np.load("labels.npy", allow_pickle=True)
num_clases = len(labels)

# Reconstruimos la red exactamente igual a como fue entrenada
model = Sequential([
    InputLayer(input_shape=(126,)),         # Entrada de datos (puntos de MediaPipe)
    Dense(256, activation='relu'),          # Capa densa 1
    Dense(128, activation='relu'),          # Capa densa 2
    Dense(64, activation='relu'),           # Capa densa 3
    Dense(num_clases, activation='softmax') # Capa final de clasificación
])

# Cargamos los pesos directamente sin dejar que Keras deserialice capas rotas
model.load_weights("modelo_senas.h5")
print("¡Etiquetas, estructura de red y pesos cargados con éxito!")

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# VARIABLES
# -----------------------------
texto = ""
ultima_letra = ""
ultimo_tiempo = time.time()
TIEMPO_ESPERA = 1.5

# Buffer predicciones
buffer_predicciones = deque(maxlen=10)

# -----------------------------
# CAMARA (Optimizado con CAP_DSHOW para Windows)
# -----------------------------
print("Iniciando la cámara web...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()

    if not success:
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    letra_actual = ""
    confianza = 0

    if results.multi_hand_landmarks:
        row = []
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])

        # Completar datos
        while len(row) < 126:
            row.extend([0, 0, 0])

        row = row[:126]
        X = np.array(row).reshape(1, 126)

        # -----------------------------
        # IA - PREDICCIÓN
        # -----------------------------
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction)
        letra_actual = labels[predicted_class]
        confianza = np.max(prediction)

        # -----------------------------
        # BUFFER
        # -----------------------------
        if confianza > 0.85:
            buffer_predicciones.append(letra_actual)

        # Verificar estabilidad
        if len(buffer_predicciones) == 10:
            letra_stable = max(
                set(buffer_predicciones),
                key=buffer_predicciones.count
            )

            repeticiones = buffer_predicciones.count(letra_stable)
            tiempo_actual = time.time()

            # Si aparece muchas veces
            if (
                repeticiones >= 8
                and letra_stable != ultima_letra
                and tiempo_actual - ultimo_tiempo > TIEMPO_ESPERA
            ):
                texto += letra_stable
                ultima_letra = letra_stable
                ultimo_tiempo = tiempo_actual
                buffer_predicciones.clear()

        # -----------------------------
        # MOSTRAR
        # -----------------------------
        cv2.putText(
            img,
            f"Letra: {letra_actual}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            img,
            f"Confianza: {confianza:.2f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # PANEL TEXTO
    # -----------------------------
    cv2.rectangle(img, (10, 400), (1000, 470), (0, 0, 0), -1)
    cv2.putText(
        img,
        f"Texto: {texto}",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    cv2.imshow("IA Traductor por Voz", img)

    # -----------------------------
    # CONTROLES
    # -----------------------------
    key = cv2.waitKey(1)

    # ESC
    if key & 0xFF == 27:
        break

    # Limpiar
    elif key == ord('c'):
        texto = ""

    # Hablar
    elif key == ord('v'):
        if texto.strip() != "":
            print(f"Reproduciendo por voz: {texto}")
            engine.say(texto)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
print("Script finalizado correctamente.")