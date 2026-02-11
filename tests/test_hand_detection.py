"""
Test Hand Detection
Run this to verify hand detection is working properly
Shows webcam feed with hand landmarks overlaid
"""


from src.recognition.hand_detection import HandDetector
import cv2
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    """Test hand detection with webcam"""
    print("="*50)
    print("HAND DETECTION TEST")
    print("="*50)
    print("\nInstructions:")
    print("1. Show your hands to the camera")
    print("2. Try different hand gestures")
    print("3. Test with one hand and two hands")
    print("4. Press 'q' to quit")
    print("\nStarting in 3 seconds...")

    import time
    time.sleep(3)

    # Initialize detector
    detector = HandDetector(max_hands=2, detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        print("\nTroubleshooting:")
        print("1. Check if webcam is connected")
        print("2. Close other apps using the webcam")
        print("3. Try: python tests/test_webcam.py")
        return False

    print("\n✓ Webcam opened successfully!")

    frame_count = 0
    hands_detected_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to grab frame")
                break

            frame_count += 1

            # Detect hands
            hands_data, annotated_frame = detector.detect_hands(frame)

            if len(hands_data) > 0:
                hands_detected_count += 1

            # Display FPS
            if frame_count > 30:
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                            (annotated_frame.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display number of hands
            cv2.putText(annotated_frame, f"Hands: {len(hands_data)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display hand details
            y_offset = 70
            for i, hand in enumerate(hands_data):
                # Hand type and confidence
                hand_text = f"{hand['handedness']}: {hand['confidence']:.2%}"
                cv2.putText(annotated_frame, hand_text,
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30

                # Extract features
                features = detector.extract_features(hand['landmarks'])
                feature_text = f"Features: {len(features)}"
                cv2.putText(annotated_frame, feature_text,
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 30

            # Instructions
            cv2.putText(annotated_frame, "Press 'q' to quit",
                        (10, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show frame
            cv2.imshow("Hand Detection Test", annotated_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    # Print statistics
    print("\n" + "="*50)
    print("TEST STATISTICS")
    print("="*50)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with hands detected: {hands_detected_count}")
    if frame_count > 0:
        detection_rate = (hands_detected_count / frame_count) * 100
        print(f"Detection rate: {detection_rate:.1f}%")

    print("\n✓ Hand detection test completed!")
    print("\nNext steps:")
    print("1. If detection works well, proceed to data collection")
    print("2. Run: python src/utils/data_collection.py --mode static")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
