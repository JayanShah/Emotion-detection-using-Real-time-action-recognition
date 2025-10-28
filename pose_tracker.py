import cv2
import mediapipe as mp

# --- 1. Initialize MediaPipe Modules ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 2. Initialize Video Capture ---
# Use 0 for the default webcam
cap = cv2.VideoCapture(0)

# --- 3. Initialize Pose Model ---
# This 'with' block ensures resources are managed correctly
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # --- 4. Main Loop (Processing Frame by Frame) ---
    while cap.isOpened():
        # Read a frame from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- 5. Process the Image ---
        # To improve performance, mark the image as not writeable
        image.flags.writeable = False
        # Convert the BGR image (OpenCV default) to RGB (MediaPipe default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make the detection
        results = pose.process(image)

        # --- 6. Draw the Skeleton ---
        # Convert the image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if any pose landmarks were detected
        if results.pose_landmarks:
            # Draw the landmarks (dots) and connections (lines)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,  # This draws the skeleton lines
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # --- 7. Display the Image ---
        # Flip the image horizontally for a natural, selfie-view display
        cv2.imshow('MediaPipe Pose Estimation', cv2.flip(image, 1))

        # Exit loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- 8. Release Resources ---
cap.release()
cv2.destroyAllWindows()