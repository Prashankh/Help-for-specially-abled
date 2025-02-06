import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the image data
DATA_DIR = './data'

# Ensure DATA_DIR exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Lists to store extracted data and corresponding labels
data = []
labels = []

# Iterate over directories in DATA_DIR
for entry in os.scandir(DATA_DIR):
    # Check if it's a directory
    if entry.is_dir():
        # Iterate over image files in each directory
        for img_path in os.listdir(entry.path):
            data_aux = []  # Temporary list to store landmarks for each image
            
            # Lists to store x and y coordinates of hand landmarks
            x_ = []
            y_ = []

            # Read image from file
            img = cv2.imread(os.path.join(entry.path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image to detect hand landmarks
            results = hands.process(img_rgb)
            
            # Extract hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract x and y coordinates of each landmark
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Calculate relative x and y coordinates and store in data_aux
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append data_aux to data and corresponding label to labels
                data.append(data_aux)
                labels.append(entry.name)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
