from function import *
from keras.models import load_model
import cv2
import numpy as np

# Load trained model
print("Loading model...")
model = load_model("model.h5", compile=False)
print("Model Loaded")

# Colors
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Probability visualization
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1
        )

        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output_frame

# Variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=1
) as hands:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Full screen detection
        image, results = mediapipe_detection(frame, hands)

        # Draw landmarks on full frame
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

            predictions.append(np.argmax(res))

            if len(predictions) >= 10:

                if np.unique(predictions[-10:])[0] == np.argmax(res):

                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:

                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])

                        else:
                            sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:]

            frame = prob_viz(res, actions, frame, colors)

        # Output bar
        cv2.rectangle(frame, (0, 0), (800, 40), (245, 117, 16), -1)

        cv2.putText(
            frame,
            "Output: " + " ".join(sentence),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()