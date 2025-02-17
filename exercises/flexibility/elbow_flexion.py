import cv2
import mediapipe as mp
import time
from exercises.utils import calculate_angle, say_async, get_exercise_metrics

def run_exercise():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    
    # Exercise tracking
    rep_count = 0
    stage = None
    start_time = time.time()
    total_paused_time = 0.0
    in_error = False
    error_start_time = None
    last_feedback_time = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Your existing pose processing code here
                landmarks = results.pose_landmarks.landmark
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Process exercise logic
                # ... (rest of your existing elbow flexion logic)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        end_time = time.time()
        cap.release()
        cv2.destroyAllWindows()
        
        return get_exercise_metrics(
            start_time, 
            end_time, 
            total_paused_time, 
            rep_count
        )
