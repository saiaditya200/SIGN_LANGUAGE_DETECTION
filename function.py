import cv2
import numpy as np
import os
import mediapipe as mp  # type: ignore

# Initialize mediapipe utilities for drawing and hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Perform mediapipe detection on an image using a specified model
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color for mediapipe
    image.flags.writeable = False  # Improve performance by disabling write access
    results = model.process(image)  # Process the image to detect hands
    image.flags.writeable = True  # Re-enable write access to the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    return image, results  # Return the processed image and detection results

# Draw landmarks and hand connections on the image
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

# Extract keypoints from the detected landmarks or return zeros if no landmarks are detected
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array(
            [[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]
        ).flatten()
        return rh
    return np.zeros(21 * 3)  # Return zero array if no hand landmarks are found

# Define paths and parameters for data collection
DATA_PATH = os.path.join('MP_Data')  # Directory for storing data
actions = np.array(['A', 'B', 'C'])  # Actions corresponding to hand gestures
no_sequences = 30  # Number of sequences for each action
sequence_length = 30  # Length of each sequence for action

# =======================
# ✅ ADDED EXECUTION BLOCK
# =======================
# This part actually runs the code using your webcam.
# Without this, the file only defines functions and exits.

if __name__ == "__main__":

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check camera opened or not
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Create resizable window
    cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)

    # Initialize MediaPipe Hands model
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Run detection
            image, results = mediapipe_detection(frame, hands)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show output
            cv2.imshow("Hand Detection", image)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()