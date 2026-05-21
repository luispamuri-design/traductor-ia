import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Puntas de dedos
tip_ids = [4, 8, 12, 16, 20]

# Cámara
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    if not success:
        break

    # Espejo
    img = cv2.flip(img, 1)

    # RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Procesar
    results = hands.process(imgRGB)

    # Si detecta manos
    if results.multi_hand_landmarks and results.multi_handedness:

        # Recorrer manos
        for hand_index, handLms in enumerate(results.multi_hand_landmarks):

            lm_list = []

            # Saber si es izquierda o derecha
            hand_label = results.multi_handedness[
                hand_index
            ].classification[0].label

            # Dibujar mano
            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            # Obtener coordenadas
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_list.append((id, cx, cy))

            fingers = []

            # -------- Pulgar --------

            if hand_label == "Right":

                if lm_list[4][1] < lm_list[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            else:

                if lm_list[4][1] > lm_list[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # -------- Otros dedos --------

            for i in range(1, 5):

                if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # Detectar gesto
            gesture = ""

            if total_fingers == 0:
                gesture = "PUNO"

            elif total_fingers == 5:
                gesture = "HOLA"

            elif total_fingers == 2:
                gesture = "PAZ"

            elif total_fingers == 1:
                gesture = "UNO"

            # Posición texto
            x = lm_list[0][1]
            y = lm_list[0][2] - 30

            # Mostrar información
            cv2.putText(
                img,
                f"{hand_label}: {gesture}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

    # Mostrar ventana
    cv2.imshow("Detector Profesional", img)

    # ESC salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()