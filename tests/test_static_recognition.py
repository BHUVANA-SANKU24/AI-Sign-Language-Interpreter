"""
Test Static Gesture Recognition
Test trained model with live webcam
"""

from src.recognition.gesture_recognition import GestureRecognizer
from src.recognition.hand_detection import HandDetector
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Test static gesture recognition"""
    print("="*60)
    print("STATIC GESTURE RECOGNITION TEST")
    print("="*60)

    # Check if model exists
    model_path = 'data/models/static_gesture_model.h5'
    encoder_path = 'data/models/static_label_encoder.npy'

    if not os.path.exists(model_path):
        print(f"\n✗ Model not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python src/recognition/train_models.py --model static")
        return

    # Initialize components
    print("\nInitializing...")
    detector = HandDetector()
    recognizer = GestureRecognizer(
        model_path, encoder_path, confidence_threshold=0.7)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return

    print("\n" + "="*60)
    print("Instructions:")
    print("  • Show hand gestures to the camera")
    print("  • System will predict the gesture in real-time")
    print("  • Press 'q' to quit")
    print("="*60)
    print("\nStarting in 3 seconds...")

    import time
    time.sleep(3)

    frame_count = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Calculate FPS
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time

        # Detect hands
        hands_data, annotated_frame = detector.detect_hands(frame)

        # Draw info box
        cv2.rectangle(annotated_frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10),
                      (400, 120), (255, 255, 255), 2)

        # Predict if hand detected
        if len(hands_data) > 0:
            # Extract features
            features = detector.extract_features(hands_data[0]['landmarks'])

            # Recognize gesture
            gesture, confidence = recognizer.recognize(features)

            if gesture:
                # Display prediction
                color = (0, 255, 0) if confidence > 0.85 else (0, 255, 255)
                cv2.putText(annotated_frame, f"Gesture: {gesture}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(annotated_frame, f"Confidence: {confidence:.1%}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show confidence bar
                bar_width = int(370 * confidence)
                cv2.rectangle(annotated_frame, (20, 90),
                              (20 + bar_width, 105), color, -1)
                cv2.rectangle(annotated_frame, (20, 90),
                              (390, 105), (255, 255, 255), 1)
            else:
                cv2.putText(annotated_frame, "Gesture: ???",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(annotated_frame, "Low confidence",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "No hand detected",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Show your hand",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS counter
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                    (annotated_frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Instructions
        cv2.putText(annotated_frame, "Press 'q' to quit",
                    (20, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Static Gesture Recognition Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\n✓ Test completed!")
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
