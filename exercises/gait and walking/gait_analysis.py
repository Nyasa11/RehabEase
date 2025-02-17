import cv2
import mediapipe as mp
import numpy as np
import time
import json
from datetime import datetime
import os
import pyttsx3
import threading
import math
import uuid
import csv

# Asynchronous voice feedback
def say_async(message):
    def _speak(msg):
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    threading.Thread(target=_speak, args=(message,)).start()


class GaitAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Enhanced session data structure for Firebase
        self.session_data = {
            'session_id': str(uuid.uuid4()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient_data': {
                'id': 'patient_' + str(uuid.uuid4())[:8],
                'session_date': datetime.now().strftime('%Y-%m-%d'),
                'session_time': datetime.now().strftime('%H:%M:%S')
            },
            'measurements': {
                'weight_distribution': [],
                'joint_angles': [],
                'balance_scores': [],
                'stride_metrics': []
            },
            'feedback_events': [],  # Stores feedback events with associated images
            'summary_metrics': {},
            'image_urls': []  # To be populated after uploading images to Firebase
        }

        # Create directory structure
        self.base_dir = 'gait_analysis_data'
        self.session_dir = os.path.join(self.base_dir, self.session_data['session_id'])
        self.images_dir = os.path.join(self.session_dir, 'images')
        self.feedback_dir = os.path.join(self.session_dir, 'feedback_images')

        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.feedback_dir, exist_ok=True)

        # Feedback tracking and periodic snapshot parameters
        self.feedback_history = []
        self.last_feedback_time = time.time()
        self.feedback_cooldown = 2.0  # seconds
        self.session_start_time = time.time()
        self.last_snapshot_time = time.time()
        self.snapshot_interval = 30  # seconds

    def save_feedback_image(self, frame, feedback_data):
        """
        Saves an image with overlaid feedback when posture issues are detected.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_id = str(uuid.uuid4())[:8]
        filename = f'feedback_{timestamp}_{image_id}.jpg'
        filepath = os.path.join(self.feedback_dir, filename)

        # Overlay text on image
        frame_with_text = frame.copy()
        y_position = 50
        for key, value in feedback_data['metrics'].items():
            text = f"{key}: {value}"
            cv2.putText(frame_with_text, text, (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_position += 30

        cv2.putText(frame_with_text, feedback_data['feedback'], (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save image
        cv2.imwrite(filepath, frame_with_text)

        # Update session data
        feedback_event = {
            'timestamp': timestamp,
            'image_id': image_id,
            'image_path': filepath,
            'feedback': feedback_data['feedback'],
            'metrics': feedback_data['metrics']
        }
        self.session_data['feedback_events'].append(feedback_event)
        return feedback_event

    def save_regular_image(self, frame):
        """
        Saves a regular snapshot of the current frame to track posture over time.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_id = str(uuid.uuid4())[:8]
        filename = f'snapshot_{timestamp}_{image_id}.jpg'
        filepath = os.path.join(self.images_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def check_posture_issues(self, metrics, landmarks):
        """
        Checks for various posture issues and returns feedback data if problems are detected.
        """
        issues = []
        critical_metrics = {}

        # Weight distribution check
        left_weight = metrics['weight_distribution']['left']
        right_weight = metrics['weight_distribution']['right']
        weight_diff = abs(left_weight - right_weight)
        if weight_diff > 20:  # threshold in percentage
            issues.append("Distribute your weight evenly")
            critical_metrics['weight_difference'] = f"{weight_diff:.1f}%"

        # Shoulder level check (to keep posture correct)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.1:
            issues.append("Keep your posture correct (shoulders level)")
            critical_metrics['shoulder_diff'] = f"{shoulder_diff:.2f}"

        # Knee angle checks (avoid over-bending or hyperextension)
        for side in ['left', 'right']:
            knee_angle = metrics['joint_angles'][f'{side}_knee']
            if knee_angle < 160:
                issues.append(f"Don't bend your {side} knee too much")
                critical_metrics[f'{side}_knee_angle'] = f"{knee_angle:.1f}°"
            elif knee_angle > 195:
                issues.append(f"Straighten your {side} knee")
                critical_metrics[f'{side}_knee_angle'] = f"{knee_angle:.1f}°"

        # Hands on thigh check
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        if abs(left_wrist.y - left_hip.y) > 0.1:
            issues.append("Keep your left hand on your thigh")
            critical_metrics['left_hand_position'] = f"{abs(left_wrist.y - left_hip.y):.2f}"
        if abs(right_wrist.y - right_hip.y) > 0.1:
            issues.append("Keep your right hand on your thigh")
            critical_metrics['right_hand_position'] = f"{abs(right_wrist.y - right_hip.y):.2f}"

        if issues:
            return {'feedback': " | ".join(issues), 'metrics': critical_metrics}
        else:
            return {'feedback': "Good posture", 'metrics': {}}

    def format_metrics_for_firebase(self, metrics):
        """
        Formats metrics into a Firebase-friendly structure.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        formatted = {
            'timestamp': timestamp,
            'weight_distribution': {
                'left': float(metrics['weight_distribution']['left']),
                'right': float(metrics['weight_distribution']['right'])
            },
            'joint_angles': {
                key: float(value)
                for key, value in metrics['joint_angles'].items()
            },
            'balance': {
                key: float(value)
                for key, value in metrics['balance'].items()
            },
            'stride_metrics': {
                key: float(value)
                for key, value in metrics['stride_metrics'].items()
            }
        }
        return formatted

    def calculate_weight_distribution(self, landmarks):
        """
        Estimates weight distribution by comparing the center-of-mass (approximated by hips)
        to the midpoint between the ankles.
        """
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x
        center_of_mass = (left_hip + right_hip) / 2

        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        midline = (left_ankle + right_ankle) / 2

        # Difference scaled to percentage: if center_of_mass is left of midline, left weight > 50%
        diff = (midline - center_of_mass) * 100
        left_weight = 50 + diff
        right_weight = 100 - left_weight

        # Clamp values between 0 and 100
        left_weight = max(0, min(100, left_weight))
        right_weight = max(0, min(100, right_weight))
        return {'left': left_weight, 'right': right_weight}

    def calculate_joint_angles(self, landmarks):
        """
        Calculates the knee angles for both legs using the cosine law.
        """
        def angle_between_points(a, b, c):
            # Calculate angle at point b given three points a, b, c
            ab = np.array([a.x - b.x, a.y - b.y])
            cb = np.array([c.x - b.x, c.y - b.y])
            dot_product = np.dot(ab, cb)
            norm_ab = np.linalg.norm(ab)
            norm_cb = np.linalg.norm(cb)
            if norm_ab * norm_cb == 0:
                return 0
            angle = math.degrees(math.acos(dot_product / (norm_ab * norm_cb)))
            return angle

        # Left knee angle
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_knee_angle = angle_between_points(left_hip, left_knee, left_ankle)

        # Right knee angle
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        right_knee_angle = angle_between_points(right_hip, right_knee, right_ankle)

        return {'left_knee': left_knee_angle, 'right_knee': right_knee_angle}

    def calculate_balance_score(self, landmarks):
        """
        Provides a simple balance score based on knee angles. The closer the knee angles
        are to 180 degrees (ideal straight leg), the higher the balance score.
        """
        joint_angles = self.calculate_joint_angles(landmarks)
        left_knee = joint_angles['left_knee']
        right_knee = joint_angles['right_knee']
        deviation = abs(left_knee - 180) + abs(right_knee - 180)
        overall_score = max(0, 100 - deviation * 0.5)
        return {'overall_score': overall_score}

    def calculate_stride_metrics(self, landmarks):
        """
        For static analysis, stride metrics are not applicable.
        Returning dummy values.
        """
        return {'stride_length': 0, 'cadence': 0}

    def display_metrics(self, frame, metrics):
        """
        Overlays the computed metrics onto the video frame.
        """
        y = 30
        for category, data in metrics.items():
            text = f"{category}: "
            if isinstance(data, dict):
                for key, value in data.items():
                    text += f"{key}={value:.1f}  "
            else:
                text += str(data)
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 25

    def save_final_session_data(self):
        """
        Computes average metrics for the session, saves session data in Firebase-friendly format,
        and writes summary metrics.
        """
        duration = time.time() - self.session_start_time

        # Calculate average weight distribution
        wd = self.session_data['measurements']['weight_distribution']
        if wd:
            avg_weight = {
                'left': np.mean([d['left'] for d in wd]),
                'right': np.mean([d['right'] for d in wd])
            }
        else:
            avg_weight = {}

        # Calculate average joint angles
        ja = self.session_data['measurements']['joint_angles']
        if ja:
            avg_joint_angles = {
                'left_knee': np.mean([d['left_knee'] for d in ja]),
                'right_knee': np.mean([d['right_knee'] for d in ja])
            }
        else:
            avg_joint_angles = {}

        # Calculate average balance score
        bs = self.session_data['measurements']['balance_scores']
        if bs:
            avg_balance = {
                'overall_score': np.mean([d['overall_score'] for d in bs])
            }
        else:
            avg_balance = {}

        # Calculate average stride metrics (dummy values)
        sm = self.session_data['measurements']['stride_metrics']
        if sm:
            avg_stride = {
                'stride_length': np.mean([d['stride_length'] for d in sm]),
                'cadence': np.mean([d['cadence'] for d in sm])
            }
        else:
            avg_stride = {}

        self.session_data['summary_metrics'] = {
            'total_frames': len(self.session_data['measurements']['weight_distribution']),
            'average_weight_distribution': avg_weight,
            'average_joint_angles': avg_joint_angles,
            'average_balance_score': avg_balance,
            'average_stride_metrics': avg_stride,
            'feedback_events_count': len(self.session_data['feedback_events']),
            'session_duration_seconds': duration
        }

        # Save session JSON
        json_filepath = os.path.join(self.session_dir, 'session_data.json')
        with open(json_filepath, 'w') as f:
            json.dump(self.session_data, f, indent=4)
        print(f"\nSession data saved to: {json_filepath}")

        print("Directory structure:")
        print(f"- Session directory: {self.session_dir}")
        print(f"- Feedback images: {self.feedback_dir}")
        print(f"- Regular images: {self.images_dir}")
        print("\nFirebase Integration Notes:")
        print("1. JSON structure is ready for Firebase Realtime Database/Firestore")
        print("2. Images can be uploaded to Firebase Storage")
        print("3. Update 'image_urls' field after uploading to Firebase Storage")

    def run_analysis(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Set the OpenCV window to full screen
        window_name = 'Gait Analysis'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        say_async("Starting gait analysis session")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Compute all measurements
                current_metrics = {
                    'weight_distribution': self.calculate_weight_distribution(landmarks),
                    'joint_angles': self.calculate_joint_angles(landmarks),
                    'balance': self.calculate_balance_score(landmarks),
                    'stride_metrics': self.calculate_stride_metrics(landmarks)
                }

                formatted_metrics = self.format_metrics_for_firebase(current_metrics)
                # Append metrics to session data
                self.session_data['measurements']['weight_distribution'].append(formatted_metrics['weight_distribution'])
                self.session_data['measurements']['joint_angles'].append(formatted_metrics['joint_angles'])
                self.session_data['measurements']['balance_scores'].append(formatted_metrics['balance'])
                self.session_data['measurements']['stride_metrics'].append(formatted_metrics['stride_metrics'])

                # Check for issues and provide feedback if needed
                current_time = time.time()
                if current_time - self.last_feedback_time >= self.feedback_cooldown:
                    feedback_data = self.check_posture_issues(current_metrics, landmarks)
                    if feedback_data and feedback_data['feedback'] != "Good posture":
                        self.save_feedback_image(frame, feedback_data)
                        say_async(feedback_data['feedback'])
                        self.last_feedback_time = current_time

                # Periodically save a regular snapshot
                if current_time - self.last_snapshot_time >= self.snapshot_interval:
                    self.save_regular_image(frame)
                    self.last_snapshot_time = current_time

                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Display real-time metrics on frame
                self.display_metrics(frame, current_metrics)

            else:
                cv2.putText(frame, "No body detected.", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.save_final_session_data()
        say_async("Analysis complete")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyzer = GaitAnalyzer()
    analyzer.run_analysis()
