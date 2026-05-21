import cv2
import mediapipe as mp
import numpy as np
import time
import os
import winsound  # El reproductor de audio más directo y nativo de Python

# Usamos pyttsx3 ÚNICAMENTE para generar el archivo de sonido guardado en disco
import pyttsx3

# Importamos las herramientas necesarias para reconstruir el modelo de forma segura
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# -----------------------------
# CARGAR IA
# -----------------------------
labels = np.load("labels.npy", allow_pickle=True)
num_clases = len(labels)

model = Sequential([
    InputLayer(input_shape=(126,)),         # Entrada de datos (puntos de MediaPipe)
    Dense(256, activation='relu'),          # Capa densa 1
    Dense(128, activation='relu'),          # Capa densa 2
    Dense(64, activation='relu'),           # Capa densa 3
    Dense(num_clases, activation='softmax') # Capa final de clasificación
])

model.load_weights("modelo_senas.h5")
print("¡Etiquetas, estructura de red y pesos cargados con éxito en traductor_voz.py!")

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# VARIABLES DE CONTROL Y TEXTO
# -----------------------------
texto = ""
ultima_letra_guardada = ""
ultimo_tiempo = time.time()
TIEMPO_ESPERA = 1.2 

# --- TU LÓGICA PERFECTA DE FOTOGRAMAS ---
historial_letras = []   
FOTOGRAMAS_VENTANA = 12 
PORCENTAJE_ACUERDO = 0.70 

# Archivo temporal para evitar conflictos de memoria
AUDIO_TEMP = "temp_output.wav"

# -----------------------------
# CAMARA
# -----------------------------
print("Iniciando la cámara web...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cámara 0 no disponible, intentando con la cámara secundaria (índice 1)...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()

    if not success:
        continue

    img = cv2.flip(img, 1)
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

        # IA - PREDICCIÓN
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction)
        letra_actual = labels[predicted_class]
        confianza = np.max(prediction)

        # -----------------------------
        # FILTRO BASADO EN HISTORIAL
        # -----------------------------
        if confianza > 0.65: 
            historial_letras.append(letra_actual)
        else:
            historial_letras.append("Incierto")

        if len(historial_letras) > FOTOGRAMAS_VENTANA:
            historial_letras.pop(0)

        # Calcular coincidencia por votación
        if len(historial_letras) == FOTOGRAMAS_VENTANA:
            letras_unicas, conteos = np.unique(historial_letras, return_counts=True)
            indice_max = np.argmax(conteos)
            letra_ganadora = letras_unicas[indice_max]
            conteo_ganador = conteos[indice_max]
            
            porcentaje_actual = conteo_ganador / FOTOGRAMAS_VENTANA
            
            if porcentaje_actual >= PORCENTAJE_ACUERDO and letra_ganadora != "Incierto":
                tiempo_actual = time.time()
                
                if (letra_ganadora != ultima_letra_guardada or (tiempo_actual - ultimo_tiempo > TIEMPO_ESPERA)):
                    texto += letra_ganadora
                    ultima_letra_guardada = letra_ganadora
                    ultimo_tiempo = tiempo_actual
                    historial_letras.clear()
            
            progreso_visual = min(int(porcentaje_actual * 100), 100)
        else:
            progreso_visual = 0

        # MOSTRAR EN PANTALLA
        cv2.putText(img, f"Viendo: {letra_actual}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, f"Estabilidad: {progreso_visual}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        historial_letras.clear()
        progreso_visual = 0

    # MOSTRAR TEXTO ACUMULADO
    cv2.rectangle(img, (10, 400), (900, 470), (0, 0, 0), -1)
    cv2.putText(img, f"Texto: {texto}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("IA Traductor por Voz", img)

    # -----------------------------
    # CONTROLES MODIFICADOS PARA EVITAR BLOQUEOS
    # -----------------------------
    key = cv2.waitKey(1)

    if key & 0xFF == 27: # ESC
        break
    elif key == 8: # Retroceso
        texto = texto[:-1]
    elif key == ord('c'): # C
        texto = ""
    elif key == ord('v'): # V
        if texto.strip() != "":
            print(f"Generando archivo de voz para: {texto}")
            try:
                # 1. Creamos un motor temporal rápido que solo escribe a disco
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                
                # Guardamos el texto hablado directamente en formato WAV
                engine.save_to_file(texto, AUDIO_TEMP)
                engine.runAndWait()
                del engine # Destruimos el motor inmediatamente para liberar memoria
                
                # 2. Winsound reproduce el WAV de forma asíncrona a bajo nivel de hardware
                # SND_FILENAME = Lee el archivo, SND_ASYNC = Corre de fondo sin congelar la cámara
                winsound.PlaySound(AUDIO_TEMP, winsound.SND_FILENAME | winsound.SND_ASYNC)
                
            except Exception as e:
                print(f"Error alternativo en reproducción: {e}")

cap.release()
cv2.destroyAllWindows()

# Limpieza al cerrar el programa
if os.path.exists(AUDIO_TEMP):
    try:
        os.remove(AUDIO_TEMP)
    except:
        pass
print("Script de voz finalizado correctamente.")