import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# -----------------------------
# CARGAR ETIQUETAS PRIMERO (Para saber cuántas clases hay)
# -----------------------------
labels = np.load("labels.npy", allow_pickle=True)
num_clases = len(labels)
print(f"¡Etiquetas cargadas con éxito! Clases detectadas: {num_clases}")


# -----------------------------
# RECONSTRUCCIÓN MANUAL EXACTA (4 CAPAS DE PESOS)
# -----------------------------
# Agregamos capas para sumar exactamente las 4 capas guardadas en tu .h5
model = Sequential([
    InputLayer(input_shape=(126,)),         # Capa de Entrada (Keras no la cuenta como capa de pesos)
    Dense(256, activation='relu'),          # Capa de pesos 1
    Dense(128, activation='relu'),          # Capa de pesos 2
    Dense(64, activation='relu'),           # Capa de pesos 3
    Dense(num_clases, activation='softmax') # Capa de pesos 4 (Salida)
])

# Cargamos solo los pesos (evita el error de incompatibilidad de load_model)
model.load_weights("modelo_senas.h5")
print("¡Estructura de red y pesos cargados con éxito!")


# -----------------------------
# CONFIGURACIÓN DE MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# CONTROL DE CÁMARA
# -----------------------------
cap = cv2.VideoCapture(0)

print("Iniciando la cámara... Presiona 'ESC' en la ventana de video para salir.")

while True:
    success, img = cap.read()
    if not success:
        print("Error: No se pudo leer la señal de la cámara web.")
        break

    # Efecto espejo
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    # Conversión a RGB para MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        row = []

        for handLms in results.multi_hand_landmarks:
            # Dibujar el esqueleto de la mano en pantalla
            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            # Extraer los puntos clave (X, Y, Z)
            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])

        # Si solo detecta una mano, rellenamos con ceros hasta completar 126 valores (63 x 2 manos)
        while len(row) < 126:
            row.extend([0, 0, 0])

        # Cortamos en caso de exceso para evitar desbordar la entrada de la red neuronal
        row = row[:126]

        # Convertimos a matriz NumPy con el formato (1, 126) para la predicción
        X = np.array(row).reshape(1, 126)

        # -----------------------------
        # PREDICCIÓN DE LA INTELIGENCIA ARTIFICIAL
        # -----------------------------
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_label = labels[predicted_class]
        confidence = np.max(prediction)

        # -----------------------------
        # DISEÑO DE LA INTERFAZ EN PANTALLA
        # -----------------------------
        cv2.rectangle(img, (10, 10), (380, 120), (0, 0, 0), -1)

        cv2.putText(
            img,
            f"Letra: {predicted_label}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.putText(
            img,
            f"Confianza: {confidence:.2f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Mostrar la ventana en tiempo real
    cv2.imshow("IA Reconocimiento de Senas", img)

    # Salir de forma segura al pulsar la tecla ESC (Escape)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Script finalizado correctamente.")