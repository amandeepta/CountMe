# pushup_counter.py

import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360-angle if angle > 180 else angle

class PushupCounter:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 up_thresh=160,
                 down_thresh=70):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.counter = 0
        self.stage = "UP"
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh

    def process(self, frame):
        """
        Process one BGR frame:
        - updates internal counter/stage
        - returns annotated frame, current count, current stage
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.pose.process(img_rgb)
        img_rgb.flags.writeable = True
        out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        try:
            lm = results.pose_landmarks.landmark
            pts = lambda l: [lm[l.value].x, lm[l.value].y]
            l_angle = calculate_angle(pts(mp_pose.PoseLandmark.LEFT_SHOULDER),
                                      pts(mp_pose.PoseLandmark.LEFT_ELBOW),
                                      pts(mp_pose.PoseLandmark.LEFT_WRIST))
            r_angle = calculate_angle(pts(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                      pts(mp_pose.PoseLandmark.RIGHT_ELBOW),
                                      pts(mp_pose.PoseLandmark.RIGHT_WRIST))
            avg = (l_angle + r_angle) / 2

            if avg > self.up_thresh:
                self.stage = "UP"
            if avg < self.down_thresh and self.stage == "UP":
                self.stage = "DOWN"
                self.counter += 1

        except Exception:
            self.stage = "Not detected"

        # overlay
        cv2.rectangle(out, (0, 0), (300, 73), (245, 110, 16), -1)
        cv2.putText(out, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(out, str(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.putText(out, 'STAGE', (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(out, self.stage, (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec((245,114,67),2,2),
                mp_drawing.DrawingSpec((245, 69,222),2,2)
            )

        return out, self.counter, self.stage

    def reset(self):
        """Reset counter and stage."""
        self.counter = 0
        self.stage = "UP"
