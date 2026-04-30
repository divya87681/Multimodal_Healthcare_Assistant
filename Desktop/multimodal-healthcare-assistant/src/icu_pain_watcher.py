import cv2
from feat import Detector

PAIN_THRESHOLD = 2.5
SKIP_FRAMES = 3

# Initialize detector (GPU if available)
detector = Detector(device='cuda')

# Start webcam
cap = cv2.VideoCapture(0)
frame_count = 0


def compute_pain_score(detected):
    au4 = detected["AU04"].values[0]   # Brow Lowerer
    au7 = detected["AU07"].values[0]   # Lid Tightener
    au10 = detected["AU10"].values[0]  # Upper Lip Raiser
    return au4 + au7 + au10


def trigger_alert(message):
    print(f"[ALERT] {message}")


while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_count += 1

    # Skip frames for performance
    if frame_count % SKIP_FRAMES != 0:
        continue

    try:
        # Convert frame to RGB for detector
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect facial features
        detected = detector.detect_image(rgb_frame)

        if detected is not None and len(detected) > 0:
            pain_score = compute_pain_score(detected)

            if pain_score > PAIN_THRESHOLD:
                trigger_alert("PAIN DETECTED")

            # Display pain score on screen
            cv2.putText(
                frame,
                f"Pain Score: {pain_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    except Exception as e:
        print("Detection error:", e)

    # Show output window
    cv2.imshow("ICU Pain Watcher", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()