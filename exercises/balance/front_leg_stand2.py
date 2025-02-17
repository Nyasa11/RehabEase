import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
import numpy as np
import queue

# Create a queue for speech messages.
speech_queue = queue.Queue()

# Create a global pyttsx3 engine instance.
engine = pyttsx3.init()

def speech_worker():
    """Continuously processes messages from the speech_queue."""
    while True:
        message = speech_queue.get()
        if message is None:  # A way to signal termination, if needed.
            break
        engine.say(message)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread.
threading.Thread(target=speech_worker, daemon=True).start()

def say_async(message):
    """Enqueue a message to be spoken by the dedicated TTS thread."""
    speech_queue.put(message)

def check_front_leg_raise_facing_camera(landmarks, leg_to_raise):
    """
    Checks if the user is correctly performing a front leg raise *facing the camera*,
    by looking at the 3D z-coordinates of the hip and ankle.
    
    Returns:
      is_correct (bool): True if posture is correct, False otherwise.
      feedback (str): A message for on-screen feedback.
      error_flags (dict): Dictionary with error types and messages.
    """
    error_flags = {}
    feedback = ""
    is_correct = True

    # Extract necessary landmarks
    left_hip    = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip   = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee   = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee  = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_shoulder  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # --- Error Check 1: Upper Body Upright ---
    if leg_to_raise == "left":
        if left_hip.y > left_knee.y:
            error_flags["upright"] = "Keep your upper body upright."
            is_correct = False
    elif leg_to_raise == "right":
        if right_hip.y > right_knee.y:
            error_flags["upright"] = "Keep your upper body upright."
            is_correct = False

    # --- Error Check 2: Shoulder Alignment ---
    if abs(left_shoulder.y - right_shoulder.y) > 0.1:
        error_flags["shoulders"] = "Keep your shoulders level."
        is_correct = False

    # Select landmarks for the active leg
    if leg_to_raise == "left":
        hip   = left_hip
        ankle = left_ankle
    elif leg_to_raise == "right":
        hip   = right_hip
        ankle = right_ankle
    else:
        error_flags["leg_select"] = "Unknown leg selection."
        return False, "Invalid leg selection.", error_flags

    # --- Error Check 3: Forward Raise via Z-Coordinate ---
    z_threshold = 0.05  # Adjust as needed
    z_diff = hip.z - ankle.z  # If ankle.z < hip.z, then z_diff > 0 means ankle is forward.

    if z_diff < z_threshold:
        error_flags["leg_raise"] = f"Raise your {leg_to_raise} leg further forward."
        is_correct = False
    else:
        feedback = f"Good job! {leg_to_raise.capitalize()} leg is forward."

    if is_correct and not feedback:
        feedback = "Posture is good."
    return is_correct, feedback, error_flags

def main():
    global mp_pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Exercise Timing Parameters
    active_duration = 10   # seconds of active (correct) hold per leg
    rest_duration = 15     # rest period increased to 15 seconds
    leg_to_raise = 'left'  # start with left leg

    # Performance Metrics
    total_active_hold = 0.0
    successful_holds = 0

    # Voice feedback control
    last_error_voice_time = 0.0
    error_voice_cooldown = 3.0

    # Active hold timer variables
    active_start_time = None
    current_active_hold = 0.0

    # For periodic positive voice feedback (every 10 seconds)
    last_positive_voice_time = 0.0
    positive_voice_interval = 10.0

    # Set up full-screen window
    window_name = 'Front Leg Raise (Facing Camera)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback_text = ""
        feedback_color = (0, 255, 0)
        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            is_correct, check_feedback, error_flags = check_front_leg_raise_facing_camera(landmarks, leg_to_raise)

            if is_correct:
                feedback_text = check_feedback
                feedback_color = (0, 255, 0)

                if active_start_time is None:
                    active_start_time = current_time
                else:
                    current_active_hold = current_time - active_start_time

                if current_time - last_positive_voice_time > positive_voice_interval:
                    say_async(f"Good job! Keep holding your {leg_to_raise} leg forward.")
                    last_positive_voice_time = current_time
            else:
                feedback_text = " ".join(error_flags.values())
                feedback_color = (0, 0, 255)

                if current_time - last_error_voice_time > error_voice_cooldown:
                    for err_msg in error_flags.values():
                        say_async(err_msg)
                    last_error_voice_time = current_time

                if active_start_time is not None:
                    total_active_hold += (current_time - active_start_time)
                    active_start_time = None
                    current_active_hold = 0.0

            hold_text = f"Hold Time: {int(current_active_hold)}s" if active_start_time else "Hold Paused"
            cv2.putText(frame, hold_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Raise your {leg_to_raise} leg forward (towards camera).", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, feedback_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)

        if active_start_time is not None and current_active_hold >= active_duration:
            say_async(f"{leg_to_raise.capitalize()} leg hold complete! Rest for {rest_duration} seconds.")
            successful_holds += 1
            total_active_hold += current_active_hold

            cv2.putText(frame, f"{leg_to_raise.capitalize()} leg hold complete!", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

            rest_start = time.time()
            while time.time() - rest_start < rest_duration:
                rest_frame = 255 * np.ones_like(frame)
                remaining_rest = int(rest_duration - (time.time() - rest_start))
                cv2.putText(rest_frame, f"Rest: {remaining_rest}s", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, rest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            leg_to_raise = "right" if leg_to_raise == "left" else "left"
            active_start_time = None
            current_active_hold = 0.0
            last_positive_voice_time = 0.0

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("===== SESSION SUMMARY =====")
    print(f"Total Active Hold Time (correct posture): {total_active_hold:.2f} seconds")
    print(f"Number of Successful Holds: {successful_holds}")

    say_async("Session ended. Well done!")

if __name__ == "__main__":
    mp_pose = mp.solutions.pose
    main()
