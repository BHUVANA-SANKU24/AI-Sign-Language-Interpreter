"""
Hand Detection Module
Uses MediaPipe to detect and track hands in real-time
Extracts hand landmarks and features for gesture recognition
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Detect and track hands using MediaPipe"""

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        """
        Initialize hand detector

        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for detection (0-1)
            tracking_confidence: Minimum confidence for tracking (0-1)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

    def detect_hands(self, frame):
        """
        Detect hands in frame and return landmarks

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            hands_data: List of dictionaries containing hand information
            annotated_frame: Frame with hand landmarks drawn
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.hands.process(frame_rgb)

        # Prepare output
        hands_data = []
        annotated_frame = frame.copy()

        # Extract hand information
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Extract 21 landmarks (x, y, z) for each hand
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                # Get hand information
                hand_info = {
                    'landmarks': np.array(landmarks),
                    # 'Left' or 'Right'
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score
                }
                hands_data.append(hand_info)

                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )

        return hands_data, annotated_frame

    def extract_features(self, landmarks):
        """
        Extract relevant features from landmarks for gesture recognition

        Args:
            landmarks: Array of 21 hand landmarks (x, y, z)

        Returns:
            features: Numpy array of extracted features
        """
        # Normalize landmarks (relative to wrist)
        wrist = landmarks[0]
        normalized = landmarks - wrist

        # Calculate palm size for scale normalization
        palm_size = np.linalg.norm(landmarks[0] - landmarks[9])
        if palm_size > 0:
            normalized = normalized / palm_size

        # Initialize features list
        features = []

        # 1. Flattened normalized coordinates (21 points * 3 coords = 63 features)
        features.extend(normalized.flatten())

        # 2. Distances between key points
        distances = self._calculate_distances(landmarks)
        features.extend(distances)

        # 3. Angles between finger segments
        angles = self._calculate_finger_angles(landmarks)
        features.extend(angles)

        return np.array(features)

    def _calculate_distances(self, landmarks):
        """Calculate distances between key landmark points"""
        distances = []

        # Fingertip to wrist distances
        fingertips = [4, 8, 12, 16, 20]  # Thumb to pinky tips
        wrist = landmarks[0]

        for tip in fingertips:
            dist = np.linalg.norm(landmarks[tip] - wrist)
            distances.append(dist)

        # Fingertip to palm center distances
        palm_center = landmarks[9]  # Middle finger base
        for tip in fingertips:
            dist = np.linalg.norm(landmarks[tip] - palm_center)
            distances.append(dist)

        # Distance between consecutive fingertips
        for i in range(len(fingertips) - 1):
            dist = np.linalg.norm(
                landmarks[fingertips[i]] - landmarks[fingertips[i+1]])
            distances.append(dist)

        return distances

    def _calculate_finger_angles(self, landmarks):
        """Calculate angles for each finger joint"""
        angles = []

        # Define finger segments: base, middle, tip
        fingers = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        for finger_name, indices in fingers.items():
            # Calculate angles between consecutive segments
            for i in range(len(indices) - 2):
                p1 = landmarks[indices[i]]
                p2 = landmarks[indices[i + 1]]
                p3 = landmarks[indices[i + 2]]

                # Vectors
                v1 = p1 - p2
                v2 = p3 - p2

                # Calculate angle
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                )
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)

        return angles

    def get_hand_bbox(self, landmarks, frame_shape):
        """
        Get bounding box for detected hand

        Args:
            landmarks: Hand landmarks
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            bbox: (x, y, w, h) bounding box
        """
        h, w = frame_shape[:2]

        # Get x, y coordinates
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h

        # Calculate bounding box
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return bbox

    def close(self):
        """Release resources"""
        self.hands.close()


def main():
    """Test hand detection with webcam"""
    print("Starting hand detection test...")
    print("Press 'q' to quit")

    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Webcam opened successfully!")
    print("Show your hands to the camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hands
        hands_data, annotated_frame = detector.detect_hands(frame)

        # Display information
        info_text = f"Hands detected: {len(hands_data)}"
        cv2.putText(annotated_frame, info_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display hand details
        y_offset = 60
        for i, hand in enumerate(hands_data):
            hand_text = f"Hand {i+1}: {hand['handedness']} ({hand['confidence']:.2f})"
            cv2.putText(annotated_frame, hand_text,
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

            # Extract and display feature count
            features = detector.extract_features(hand['landmarks'])
            feature_text = f"  Features: {len(features)}"
            cv2.putText(annotated_frame, feature_text,
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25

        cv2.putText(annotated_frame, "Press 'q' to quit",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("Hand Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Test completed!")


if __name__ == "__main__":
    main()
