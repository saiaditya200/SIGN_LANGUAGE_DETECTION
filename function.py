import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize mediapipe utilities for drawing and hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Perform mediapipe detection on an image using a specified model
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Draw landmarks
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

# Extract keypoints
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array(
            [[res.x, res.y, res.z]
             for res in results.multi_hand_landmarks[0].landmark]
        ).flatten()
        return rh
    return np.zeros(63)

DATA_PATH = os.path.join("MP_Data")
actions = np.array(["A", "B", "C"])
no_sequences = 30
sequence_length = 30

print("Program Started")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, hands)

        draw_styled_landmarks(image, results)

        cv2.imshow("Hand Detection", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()