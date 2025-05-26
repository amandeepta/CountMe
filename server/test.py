import cv2
import mediapipe as mp
from pushup import PushupCounter
from squats import SquatCounter

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(1)

    pushup_counter = PushupCounter()
    squat_counter = SquatCounter()

    workout_mode = 'pushups'  # toggle with 'w'

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False

            results = pose.process(img_rgb)
            img_rgb.flags.writeable = True
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if workout_mode == 'pushups':
                    count, angle, horiz = pushup_counter.update(landmarks)
                    cv2.putText(img, f'Pushups: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                    cv2.putText(img, f'Elbow Angle: {angle}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(img, f'Body Horizontal: {"Yes" if horiz else "No"}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                elif workout_mode == 'squats':
                    count, angle, vert = squat_counter.update(landmarks)
                    cv2.putText(img, f'Squats: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                    cv2.putText(img, f'Knee Angle: {angle}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(img, f'Body Vertical: {"Yes" if vert else "No"}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(img, f'Mode: {workout_mode}', (30, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow('Workout Counter', img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                workout_mode = 'squats' if workout_mode == 'pushups' else 'pushups'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
