import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque, Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pushup import PushupCounter
from squats import SquatCounter

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

class FeatureSmoother:
    def __init__(self, window=5):
        self.window = window
        self.buffers = {}
    def smooth(self, feat_dict):
        smoothed = {}
        for k, v in feat_dict.items():
            buf = self.buffers.setdefault(k, deque(maxlen=self.window))
            buf.append(v)
            smoothed[k] = sum(buf)/len(buf)
        return smoothed

df = pd.read_csv('../exercise_angles.csv')
df['Side'] = df['Side'].map({'left': 0, 'right': 1})
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))
feature_order = list(X.columns)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

smoother = FeatureSmoother(window=5)
label_buffer = deque(maxlen=10)
PROB_THRES = 0.6

cap = cv2.VideoCapture(1)
pushup_counter = PushupCounter()
squat_counter = SquatCounter()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    pred_label = "unknown"
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        lv = lm[11].visibility + lm[13].visibility + lm[15].visibility
        rv = lm[12].visibility + lm[14].visibility + lm[16].visibility
        side = 'left' if lv > rv else 'right'
        idx = {
            'left':  {'shoulder':11,'elbow':13,'wrist':15,'hip':23,'knee':25,'ankle':27},
            'right': {'shoulder':12,'elbow':14,'wrist':16,'hip':24,'knee':26,'ankle':28}
        }[side]
        coords = {k:[lm[i].x, lm[i].y] for k,i in idx.items()}
        vert = [coords['shoulder'][0], 1.0]
        raw = {
            'Side': 0 if side=='left' else 1,
            'Shoulder_Angle': calculate_angle(coords['elbow'], coords['shoulder'], coords['hip']),
            'Elbow_Angle':    calculate_angle(coords['shoulder'], coords['elbow'], coords['wrist']),
            'Hip_Angle':      calculate_angle(coords['shoulder'], coords['hip'], coords['knee']),
            'Knee_Angle':     calculate_angle(coords['hip'], coords['knee'], coords['ankle']),
            'Ankle_Angle':    calculate_angle(coords['knee'], coords['ankle'], [coords['ankle'][0],coords['ankle'][1]+0.1]),
            'Shoulder_Ground_Angle': calculate_angle(vert, coords['shoulder'], coords['elbow']),
            'Elbow_Ground_Angle':    calculate_angle(vert, coords['elbow'], coords['wrist']),
            'Hip_Ground_Angle':      calculate_angle(vert, coords['hip'], coords['knee']),
            'Knee_Ground_Angle':     calculate_angle(vert, coords['knee'], coords['ankle']),
            'Ankle_Ground_Angle':    90.0
        }
        feats = smoother.smooth(raw)
        if (feats['Knee_Angle']>165 and feats['Hip_Angle']>165 and
            feats['Elbow_Angle']>160 and abs(feats['Shoulder_Ground_Angle']-90)<15):
            pred_label = "standing"
        else:
            X_live = np.array([[feats[f] for f in feature_order]])
            probs = model.predict_proba(X_live)[0]
            top_idx = np.argmax(probs)
            pred_label = model.classes_[top_idx] if probs[top_idx] >= PROB_THRES else "standing"
    label_buffer.append(pred_label)
    pred = Counter(label_buffer).most_common(1)[0][0]
    print(pred)
    if pred == "Push Ups":
        annotated_frame, count, stage = pushup_counter.process(frame)
        frame = annotated_frame
    elif pred == "Squats" :
        annotated_frame, reps = squat_counter.process(frame)
        frame = annotated_frame
    else:
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f'Exercise: {pred}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Detector", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
