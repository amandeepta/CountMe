import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Helper ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# === Load and train model ===
df = pd.read_csv('../exercise_angles.csv')
df['Side'] = df['Side'].map({'left': 0, 'right': 1})

X = df.drop(columns=['Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save feature column order
feature_order = list(X.columns)

# === Mediapipe setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Auto-detect better visible side
        left_visibility = lm[11].visibility + lm[13].visibility + lm[15].visibility
        right_visibility = lm[12].visibility + lm[14].visibility + lm[16].visibility
        side = 'left' if left_visibility > right_visibility else 'right'

        idx = {
            'left':  {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27},
            'right': {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28}
        }[side]

        coords = {k: [lm[i].x, lm[i].y] for k, i in idx.items()}
        vertical = [coords['shoulder'][0], 1.0]  # vertical reference

        row = {
            'Side': 0 if side == 'left' else 1,
            'Shoulder_Angle': calculate_angle(coords['elbow'], coords['shoulder'], coords['hip']),
            'Elbow_Angle': calculate_angle(coords['shoulder'], coords['elbow'], coords['wrist']),
            'Hip_Angle': calculate_angle(coords['shoulder'], coords['hip'], coords['knee']),
            'Knee_Angle': calculate_angle(coords['hip'], coords['knee'], coords['ankle']),
            'Ankle_Angle': calculate_angle(coords['knee'], coords['ankle'], [coords['ankle'][0], coords['ankle'][1] + 0.1]),
            'Shoulder_Ground_Angle': calculate_angle(vertical, coords['shoulder'], coords['elbow']),
            'Elbow_Ground_Angle': calculate_angle(vertical, coords['elbow'], coords['wrist']),
            'Hip_Ground_Angle': calculate_angle(vertical, coords['hip'], coords['knee']),
            'Knee_Ground_Angle': calculate_angle(vertical, coords['knee'], coords['ankle']),
            'Ankle_Ground_Angle': 90.0
        }

        # Ensure correct order
        X_live = np.array([[row[feat] for feat in feature_order]])
        pred = model.predict(X_live)[0]

        # Draw pose and prediction
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f'Exercise: {pred}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Debug info (optional)
        # print(pd.DataFrame([row]))  # Uncomment to debug angle values

    cv2.imshow("Exercise Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
