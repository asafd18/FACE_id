import os
import cv2
import numpy as np
from mediapipe import solutions as mp_solutions
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# הגדרת MediaPipe Hands
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# תיקיית הנתונים
DATA_DIR = './sign_language_data'
MODEL_PATH = './sign_language_model.pkl'
SCALER_PATH = './sign_language_scaler.pkl'

def extract_hand_landmarks(image):
    """חילוץ נקודות מפתח של היד מתמונה"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None

def load_data():
    """טעינת נתונים מתיקיית sign_language_data"""
    X, y = [], []
    for letter in os.listdir(DATA_DIR):
        letter_dir = os.path.join(DATA_DIR, letter)
        if os.path.isdir(letter_dir):
            for img_file in os.listdir(letter_dir):
                img_path = os.path.join(letter_dir, img_file)
                img = cv2.imread(img_path)
                landmarks = extract_hand_landmarks(img)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(letter)
    return np.array(X), np.array(y)

def train_model():
    """אימון מודל סיווג"""
    X, y = load_data()
    if len(X) == 0:
        print("No data found. Please collect data first.")
        return

    # סטנדרטיזציה של הנתונים
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # אימון מודל Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # שמירת המודל והסקיילר
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_model()