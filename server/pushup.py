import numpy as np
from collections import deque
import mediapipe as mp

class PushupCounter:
    def __init__(self):
        self.pushup_count = 0
        self.stage = None
        self.elbow_angles = deque(maxlen=5)  # smoother with fixed maxlen

    def calculate_angle(self, a, b, c):
        # a, b, c are 3D points [x,y,z]
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def is_body_horizontal(self, shoulder, hip, knee, threshold=20):
        # Check if body is roughly horizontal by angle between shoulder-hip-knee
        angle = self.calculate_angle(shoulder, hip, knee)
        return (90 - threshold) < angle < (90 + threshold)

    def smooth_angle(self, new_angle):
        self.elbow_angles.append(new_angle)
        return np.mean(self.elbow_angles)

    def update(self, landmarks):
        mp_pose = mp.solutions.pose

        # Extract 3D coordinates for right side keypoints
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]

        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]

        # Calculate elbow angle and smooth it
        raw_elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        elbow_angle = self.smooth_angle(raw_elbow_angle)

        # Check if body is horizontal enough to consider pushup movement valid
        horizontal = self.is_body_horizontal(shoulder, hip, knee)

        # Pushup counting logic
        if horizontal:
            if self.stage in [None, 'up'] and elbow_angle < 90:
                self.stage = 'down'
            elif self.stage == 'down' and elbow_angle > 160:
                self.stage = 'up'
                self.pushup_count += 1
        else:
            # Reset stage if body is not horizontal (prevents false counts)
            self.stage = None

        return self.pushup_count, int(elbow_angle), horizontal
