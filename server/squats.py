import numpy as np
from collections import deque
import mediapipe as mp

class SquatCounter:
    def __init__(self):
        self.squat_count = 0
        self.stage = None
        self.knee_angles = deque(maxlen=7)  # Slightly larger smoothing window
        self.stage_hold_frames = 0
        self.min_hold_frames = 5  # Require pose to be stable for 5 frames

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def median_angle(self, new_angle):
        self.knee_angles.append(new_angle)
        return np.median(self.knee_angles)

    def torso_angle(self, shoulder, hip, knee):
        # Angle at hip between shoulder-hip-knee (measures torso lean)
        return self.calculate_angle(shoulder, hip, knee)

    def update(self, landmarks):
        mp_pose = mp.solutions.pose
        
        # Extract key landmarks for right and left
        def get_point(lm, idx):
            p = lm[idx]
            return [p.x, p.y, p.z]

        r_shoulder = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        r_hip = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
        r_knee = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
        r_ankle = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        l_shoulder = get_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        l_hip = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
        l_knee = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
        l_ankle = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)

        # Calculate knee angles and torso angles both sides
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        avg_knee_angle = (r_knee_angle + l_knee_angle) / 2

        r_torso_angle = self.torso_angle(r_shoulder, r_hip, r_knee)
        l_torso_angle = self.torso_angle(l_shoulder, l_hip, l_knee)
        avg_torso_angle = (r_torso_angle + l_torso_angle) / 2

        # Median smooth knee angle to reduce noise spikes
        knee_angle = self.median_angle(avg_knee_angle)

        # Vertical position checks:
        # 1) Hip above knee (y-axis)
        hip_y = (r_hip[1] + l_hip[1]) / 2
        knee_y = (r_knee[1] + l_knee[1]) / 2
        ankle_y = (r_ankle[1] + l_ankle[1]) / 2

        hip_above_knee = hip_y < knee_y + 0.1
        hip_above_ankle = hip_y < ankle_y + 0.2

        # Torso should be roughly vertical (angle close to 180 degrees)
        torso_upright = avg_torso_angle > 140

        # Confirm correct squat posture
        posture_ok = hip_above_knee and hip_above_ankle and torso_upright

        # Squat counting with hold frame hysteresis to avoid flickering
        if posture_ok:
            if self.stage in [None, 'up'] and knee_angle < 110:
                self.stage_hold_frames += 1
                if self.stage_hold_frames >= self.min_hold_frames:
                    self.stage = 'down'
                    self.stage_hold_frames = 0
            elif self.stage == 'down' and knee_angle > 150:
                self.stage_hold_frames += 1
                if self.stage_hold_frames >= self.min_hold_frames:
                    self.stage = 'up'
                    self.squat_count += 1
                    self.stage_hold_frames = 0
            else:
                self.stage_hold_frames = 0
        else:
            self.stage = None
            self.stage_hold_frames = 0

        return self.squat_count, int(knee_angle), posture_ok
