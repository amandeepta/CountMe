import cv2
import mediapipe as mp
from pushup import PushupCounter
from squats import SquatCounter

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(1)  # Use 0 for default webcam

    pushup_counter = PushupCounter()
    squat_counter = SquatCounter()

    workout_mode = 'pushups'  # Press 'w' to switch

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

            count = 0
            feedback = ''

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if workout_mode == 'pushups':
                    count, feedback, completed = pushup_counter.update(landmarks)
                elif workout_mode == 'squats':
                    count, feedback, completed = squat_counter.update(landmarks)

                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Display repetition count
            label = f'Pushups: {count}' if workout_mode == 'pushups' else f'Squats: {count}'
            cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 4, cv2.LINE_AA)

            # Display feedback message if available
            if feedback:
                print("Feedback:", feedback)

                cv2.putText(img, str(feedback) if feedback is not None else "", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


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
