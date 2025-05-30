# squat_counter.py

import cv2
import numpy as np
import mediapipe as mp

def _find_angle(a, b, c, min_vis=0.8):
    if a.visibility > min_vis and b.visibility > min_vis and c.visibility > min_vis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return 360 - angle if angle > 180 else angle
    return -1.0

def _leg_state(angle):
    if angle < 0:
        return 0
    return 1 if angle <= 70 else 2

class SquatCounter:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing = mp.solutions.drawing_utils
        self.rep_count = 0
        self._state = 0

    def process(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            r_angle = _find_angle(lm[24], lm[26], lm[28])
            l_angle = _find_angle(lm[23], lm[25], lm[27])
            r_state = _leg_state(r_angle)
            l_state = _leg_state(l_angle)

            combined = 0
            if r_state == 1 and l_state == 1:
                combined = 1
            elif r_state == 2 and l_state == 2:
                combined = 2

            if self._state == 0 and combined == 2:
                self._state = 2
            elif self._state == 2 and combined == 1:
                self._state = 1
            elif self._state == 1 and combined == 2:
                self.rep_count += 1
                self._state = 2

            self.drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                self.drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                self.drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            fb = []
            if r_state == 0:
                fb.append("Right leg not detected")
            if l_state == 0:
                fb.append("Left leg not detected")
            if not fb:
                if combined == 1:
                    fb = ["Squatting"]
                elif combined == 2:
                    fb = ["Standing"]
                else:
                    fb = ["Uneven leg positions"]
            msg = ", ".join(fb)

            cv2.putText(frame, f'Squats: {self.rep_count}', (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, msg, (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        return frame, self.rep_count

    def reset(self):
        self.rep_count = 0
        self._state = 0
