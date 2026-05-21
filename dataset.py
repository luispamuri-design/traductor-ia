import cv2
import mediapipe as mp
import pandas as pd
import os

# ---------- CONFIGURACION ----------

LABEL = "Z"

TOTAL_MUESTRAS = 300

ARCHIVO = "dataset.csv"

# -----------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []

contador = 0

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

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

        # Completar si solo hay una mano
        while len(row) < 126:
            row.extend([0, 0, 0])

        row.append(LABEL)

        data.append(row)

        contador += 1

        cv2.putText(
            img,
            f"Muestras: {contador}/{TOTAL_MUESTRAS}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow("Capturando Dataset", img)

    if contador >= TOTAL_MUESTRAS:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Guardar CSV

columnas = []

for hand in range(2):
    for point in range(21):
        columnas += [
            f"x_{hand}_{point}",
            f"y_{hand}_{point}",
            f"z_{hand}_{point}"
        ]

columnas.append("label")

df = pd.DataFrame(data, columns=columnas)

# Crear o agregar archivo

if os.path.exists(ARCHIVO):

    viejo = pd.read_csv(ARCHIVO)

    df = pd.concat([viejo, df], ignore_index=True)

df.to_csv(ARCHIVO, index=False)

print("Dataset guardado correctamente")