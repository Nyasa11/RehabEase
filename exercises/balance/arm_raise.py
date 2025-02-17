import cv2
import mediapipe as mp
import time
import pyttsx3
import threading

# -----------------------------
# Voice Feedback Setup (TTS)
# -----------------------------
engine = pyttsx3.init()

def say(message):
    engine.say(message)
    engine.runAndWait()

def say_async(message):
    # Run the speech in a separate thread so it doesn't block processing.
    threading.Thread(target=lambda: say(message)).start()

# -----------------------------
# Exercise Parameters & States
# -----------------------------
required_hold_duration = 5.0  # seconds required to hold arms overhead
posture_threshold = 0.1       # normalized threshold for shoulder alignment

state = "waiting"  # possible states: "waiting", "holding", "repComplete"
hold_start_time = None
total_reps = 0
posture_errors = 0

session_start_time = None
last_feedback_time = 0
feedback_cooldown = 3.0  # seconds

# -----------------------------
# MediaPipe Pose Initialization
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Initialize Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a named window and set it to full screen.
window_name = "Overhead Arm Raises"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Allow the camera to warm up.
time.sleep(2)
session_start_time = time.time()
say_async("Welcome! Please stand back so I can see your full body for the overhead arm raises exercise.")

# -----------------------------
# Main Loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect.
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose.
    results = pose.process(image_rgb)

    # Draw landmarks if a pose is detected.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    current_time = time.time()
    elapsed_time = current_time - session_start_time
    main_feedback = "No pose detected"  # Default feedback.

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Key landmarks:
        # Nose (0), Left Shoulder (11), Right Shoulder (12),
        # Left Wrist (15), Right Wrist (16)
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # Check if arms are raised (both wrists above the nose).
        arms_raised = (left_wrist.y < nose.y) and (right_wrist.y < nose.y)

        # Check if shoulders are level.
        if abs(left_shoulder.y - right_shoulder.y) > posture_threshold:
            if (current_time - last_feedback_time) > feedback_cooldown:
                say_async("Keep your shoulders level")
                last_feedback_time = current_time
            posture_errors += 1
            main_feedback = "Adjust shoulder alignment"
        else:
            # State Machine for Exercise Logic.
            if state == "waiting":
                main_feedback = "Raise your arms overhead"
                if arms_raised:
                    state = "holding"
                    hold_start_time = current_time
                    say_async("Good, hold that position")
            elif state == "holding":
                if not arms_raised:
                    state = "waiting"
                    hold_start_time = None
                    say_async("Arms dropped. Please raise them overhead again")
                    main_feedback = "Raise your arms overhead"
                else:
                    hold_time = current_time - hold_start_time
                    remaining = max(0, required_hold_duration - hold_time)
                    main_feedback = f"Hold for {remaining:.1f} sec"
                    if hold_time >= required_hold_duration:
                        total_reps += 1
                        say_async("Great job! Now lower your arms")
                        state = "repComplete"
            elif state == "repComplete":
                main_feedback = "Lower your arms to reset"
                if not arms_raised:
                    state = "waiting"
    else:
        main_feedback = "No pose detected"

    # -----------------------------
    # Overlay Feedback & Stats
    # -----------------------------
    cv2.putText(frame, main_feedback, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    stats_text = (f"Total Reps: {total_reps}    "
                  f"Session Time: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}    "
                  f"Posture Errors: {posture_errors}")
    cv2.putText(frame, stats_text, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the full-screen window.
    cv2.imshow(window_name, frame)

    # Press 'q' to end the session.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# End of Session
# -----------------------------
total_session_time = time.time() - session_start_time
completion_message = (f"Session Complete!\n"
                      f"Total Reps: {total_reps}\n"
                      f"Session Time: {int(total_session_time // 60)}:{int(total_session_time % 60):02d}\n"
                      f"Posture Errors: {posture_errors}")
print(completion_message)
say_async("Great workout! You've completed your session successfully!")

cap.release()
cv2.destroyAllWindows()
