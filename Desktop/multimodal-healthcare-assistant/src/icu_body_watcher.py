import cv2
import mediapipe as mp
import math

ARM_THRESH = 25
HEAD_THRESH = 15
AGITATION_LIMIT = 20
DECAY = 1

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

prev_nose = None
prev_l_wrist = None
prev_r_wrist = None

agitation_counter = 0


def calculate_movement(curr, prev):
    if curr is None or prev is None:
        return 0
    return math.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        nose = (
            landmarks[mp_pose.PoseLandmark.NOSE].x,
            landmarks[mp_pose.PoseLandmark.NOSE].y
        )

        l_wrist = (
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        )

        r_wrist = (
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        )

        # Calculate movement speeds
        head_speed = calculate_movement(nose, prev_nose)
        l_speed = calculate_movement(l_wrist, prev_l_wrist)
        r_speed = calculate_movement(r_wrist, prev_r_wrist)

        arm_speed = max(l_speed, r_speed)

        # Agitation logic
        if arm_speed > ARM_THRESH:
            agitation_counter += 2
        elif head_speed > HEAD_THRESH:
            agitation_counter += 1
        else:
            agitation_counter = max(0, agitation_counter - DECAY)

        # Update previous positions
        prev_nose = nose
        prev_l_wrist = l_wrist
        prev_r_wrist = r_wrist

        # Status display
        status_text = "NORMAL"
        color = (0, 255, 0)

        if agitation_counter >= AGITATION_LIMIT:
            status_text = "CRITICAL: PATIENT THRASHING"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Show output
    cv2.imshow("ICU Agitation Monitor", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()