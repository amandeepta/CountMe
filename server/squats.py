import sys
import mediapipe as mp
import cv2
import numpy as np

def findAngle(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * (180 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle
    else:
        return -1

def legState(angle):
    if angle < 0:
        return 0  # Not detected
    elif angle <= 70:
        return 1  # Squat (bent)
    else:
        return 2  # Standing (extended)

def feedbackMessage(rState, lState):
    if rState == 0 or lState == 0:
        msgs = []
        if rState == 0:
            msgs.append("Right leg not detected")
        if lState == 0:
            msgs.append("Left leg not detected")
        return ", ".join(msgs)
    if rState == 1 and lState == 1:
        return "Squatting"
    if rState == 2 and lState == 2:
        return "Standing"
    return "Uneven leg positions"

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if len(sys.argv) < 2:
        cap = cv2.VideoCapture(1)  # Webcam index 0
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            sys.exit(1)

    repCount = 0

    # States for squat detection:
    # 0 = waiting to start (assume standing)
    # 1 = squat detected (down)
    # 2 = standing detected after squat (up)
    state = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1024, 600))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if results.pose_landmarks:
                lm_arr = results.pose_landmarks.landmark

                rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

                rState = legState(rAngle)
                lState = legState(lAngle)

                # Only consider squat if both legs are detected and consistent
                if rState != 0 and lState != 0:
                    # Determine combined state: 1 if both squat, 2 if both standing, else 0 (uneven)
                    combinedState = 0
                    if rState == 1 and lState == 1:
                        combinedState = 1  # Squat down
                    elif rState == 2 and lState == 2:
                        combinedState = 2  # Standing up

                    # State machine for counting reps
                    if state == 0 and combinedState == 2:
                        # Starting position is standing
                        state = 2
                    elif state == 2 and combinedState == 1:
                        # Went down into squat
                        state = 1
                    elif state == 1 and combinedState == 2:
                        # Back up after squat: count 1 rep
                        repCount += 1
                        state = 2

                fb = feedbackMessage(rState, lState)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                cv2.putText(frame, f'Squats: {repCount}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, fb, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Squat Rep Counter", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
