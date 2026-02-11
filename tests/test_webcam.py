"""
Test Webcam
Simple script to test if your webcam is working
Press 'q' to quit
"""

import cv2
import sys


def test_webcam(camera_id=0):
    """Test webcam functionality"""
    print("="*60)
    print("WEBCAM TEST")
    print("="*60)
    print(f"\nAttempting to open camera {camera_id}...")

    # Try to open camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"✗ Could not open camera {camera_id}")
        print("\nTroubleshooting:")
        print("1. Check if camera is connected")
        print("2. Check camera permissions")
        print("3. Try different camera ID:")
        print("   python tests/test_webcam.py --camera 1")
        print("4. Close other apps using the camera")
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print("✓ Camera opened successfully!")
    print(f"\nCamera Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print("\nShowing camera feed...")
    print("Press 'q' to quit")
    print("-"*60)

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("✗ Failed to grab frame")
                break

            frame_count += 1

            # Add frame counter and info
            cv2.putText(frame, f"Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Resolution: {width}x{height}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, "Press 'q' to quit",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw a simple border
            cv2.rectangle(frame, (5, 5), (width-5, height-5), (0, 255, 0), 2)

            # Show frame
            cv2.imshow("Webcam Test", frame)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nUser pressed 'q' to quit")
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"webcam_test_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("✓ Webcam test completed successfully!")
    print(f"Total frames captured: {frame_count}")
    print("="*60)
    print("\nNext step: Test hand detection")
    print("Run: python tests/test_hand_detection.py")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test webcam functionality')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID (default: 0 for built-in camera, try 1 for external)')
    args = parser.parse_args()

    success = test_webcam(args.camera)

    sys.exit(0 if success else 1)
