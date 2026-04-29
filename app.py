from function import *
from keras.models import load_model
import cv2
import numpy as np

# Load model
print("Loading model...")
model = load_model("model.h5", compile=False)
print("Model Loaded")

# Labels
labels = actions

# Colors
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 165, 255)
]

# Prediction threshold
threshold = 0.6

# Show probability bars
def prob_viz(res, labels, frame):
    output = frame.copy()

    for i in range(len(labels)):
        prob = float(res[i])

        cv2.rectangle(
            output,
            (0, 60 + i * 50),
            (int(prob * 250), 100 + i * 50),
            colors[i],
            -1
        )

        cv2.putText(
            output,
            labels[i] + " : " + str(round(prob * 100, 1)) + "%",
            (10, 92 + i * 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output

# Variables
sequence = []
predictions = []
sentence = []

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Hand detection
        image, results = mediapipe_detection(frame, hands)

        # Draw landmarks
        draw_styled_landmarks(frame, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Predict after 30 frames
        if len(sequence) == 30:

            res = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
            )[0]

            pred = int(np.argmax(res))
            confidence = float(res[pred])

            predictions.append(pred)
            predictions = predictions[-3:]

            # Stable last 3 predictions same
            if len(predictions) == 3:

                if predictions[0] == predictions[1] == predictions[2]:

                    if confidence > threshold:

                        word = labels[pred]

                        if len(sentence) == 0:
                            sentence.append(word)

                        elif word != sentence[-1]:
                            sentence.append(word)

            # Keep only latest output
            sentence = sentence[-1:]

            # Draw bars
            frame = prob_viz(res, labels, frame)

        # Top output bar
        cv2.rectangle(
            frame,
            (0, 0),
            (900, 50),
            (50, 50, 200),
            -1
        )

        cv2.putText(
            frame,
            "Output: " + " ".join(sentence),
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Sign Language Detection", frame)

        # Quit
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()