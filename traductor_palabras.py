import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# -----------------------------
# CARGAR IA CORREGIDO
# -----------------------------
labels = np.load("labels.npy", allow_pickle=True)
num_clases = len(labels)

model = Sequential([
    InputLayer(input_shape=(126,)),         # Entrada
    Dense(256, activation='relu'),          # Capa de pesos 1
    Dense(128, activation='relu'),          # Capa de pesos 2
    Dense(64, activation='relu'),           # Capa de pesos 3
    Dense(num_clases, activation='softmax') # Capa de pesos 4 (Salida)
])

model.load_weights("modelo_senas.h5")
print("¡Etiquetas, estructura de red y pesos cargados con éxito!")

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# VARIABLES
# -----------------------------
texto = ""
ultima_letra = ""
ultimo_tiempo = time.time()
TIEMPO_ESPERA = 1.5

# -----------------------------
# CONTROL DE CÁMARA (Optimizado para Windows)
# -----------------------------
print("Intentando conectar con la cámara web...")

# Usamos el índice 0 con CAP_DSHOW para evitar congelamientos en Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Si la cámara 0 no abre, intentamos con la 1 (útil si usas DroidCam o cámaras virtuales)
if not cap.isOpened():
    print("Cámara 0 no disponible, intentando con cámara 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("¡ERROR CRÍTICO!: No se detectó ninguna cámara web conectada o está siendo usada por otra app (como Discord, Zoom o Teams).")
else:
    print("¡Cámara encendida con éxito! Iniciando bucle de video...")

print("Presiona la tecla 'ESC' en la ventana de video para salir.")

while True:
    success, img = cap.read()

    if not success:
        # Imprimimos un aviso en consola si hay un pestañeo en la señal de la cámara
        print("Esperando señal de video válida...")
        time.sleep(0.1)
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    letra_actual = ""

    if results.multi_hand_landmarks:
        row = []
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])

        while len(row) < 126:
            row.extend([0, 0, 0])

        row = row[:126]
        X = np.array(row).reshape(1, 126)

        # Predicción
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction)
        letra_actual = labels[predicted_class]
        confianza = np.max(prediction)

        # Agregar letra al texto
        tiempo_actual = time.time()
        if (letra_actual != ultima_letra and confianza > 0.80 and tiempo_actual - ultimo_tiempo > TIEMPO_ESPERA):
            texto += letra_actual
            ultima_letra = letra_actual
            ultimo_tiempo = tiempo_actual

        # Interfaz de letras en pantalla
        cv2.putText(img, f"Letra: {letra_actual}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, f"Confianza: {confianza:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostrar la barra inferior de texto acumulado
    cv2.rectangle(img, (10, 400), (900, 470), (0, 0, 0), -1)
    cv2.putText(img, f"Texto: {texto}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Mostrar ventana gráfica
    cv2.imshow("IA Traductor de Palabras", img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27: # ESC salir
        break
    elif key == 8: # BACKSPACE borrar última letra
        texto = texto[:-1]
    elif key == ord('c'): # C limpiar todo
        texto = ""

cap.release()
cv2.destroyAllWindows()
print("Script finalizado correctamente.")