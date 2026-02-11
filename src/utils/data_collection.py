"""
Data Collection Module
Collect sign language gesture data using webcam
Supports both static gestures (letters) and dynamic gestures (words)
"""

from src.recognition.hand_detection import HandDetector
import cv2
import os
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DataCollector:
    """Collect gesture data using webcam"""

    def __init__(self, save_dir='data/raw'):
        """
        Initialize data collector

        Args:
            save_dir: Directory to save collected data
        """
        self.save_dir = save_dir
        self.detector = HandDetector(max_hands=2)

        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)

    def collect_static_gestures(self, class_name, num_samples=100):
        """
        Collect static gesture data (images)

        Args:
            class_name: Name of the gesture class (e.g., 'A', 'B', 'hello')
            num_samples: Number of samples to collect
        """
        class_dir = os.path.join(self.save_dir, 'static', class_name)
        os.makedirs(class_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return

        count = 0
        print("\n" + "="*60)
        print(f"COLLECTING STATIC GESTURE: '{class_name}'")
        print("="*60)
        print(f"Target: {num_samples} samples")
        print("\nInstructions:")
        print("1. Position your hand to show the gesture")
        print("2. Press 's' to save the current frame")
        print("3. Press 'q' to finish early")
        print("4. Try to vary:")
        print("   - Hand position (left, right, center)")
        print("   - Distance from camera (close, far)")
        print("   - Hand orientation (slight rotations)")
        print("   - Lighting (if possible)")
        print("\nStarting in 3 seconds...")
        time.sleep(3)

        last_save_time = 0
        save_cooldown = 0.5  # Prevent accidental multiple saves

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect hands
            hands_data, annotated_frame = self.detector.detect_hands(frame)

            # Add collection interface
            # Title bar
            cv2.rectangle(annotated_frame, (0, 0),
                          (annotated_frame.shape[1], 100), (0, 0, 0), -1)

            # Progress
            progress = (count / num_samples) * 100
            progress_bar_width = int(
                (annotated_frame.shape[1] - 40) * (count / num_samples))
            cv2.rectangle(annotated_frame, (20, 80),
                          (20 + progress_bar_width, 95), (0, 255, 0), -1)

            # Text information
            cv2.putText(annotated_frame, f"Gesture: {class_name}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Collected: {count}/{num_samples} ({progress:.1f}%)",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Hand detection status
            if len(hands_data) > 0:
                status_color = (0, 255, 0)
                status_text = "Hand detected - Press 's' to save"
            else:
                status_color = (0, 0, 255)
                status_text = "No hand detected"

            cv2.putText(annotated_frame, status_text,
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Instructions
            cv2.putText(annotated_frame, "Controls: 's'=save | 'q'=quit",
                        (20, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Data Collection - Static Gestures", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()

            if key == ord('s') and len(hands_data) > 0:
                # Check cooldown
                if current_time - last_save_time > save_cooldown:
                    # Save image
                    img_path = os.path.join(
                        class_dir, f"{class_name}_{count:04d}.jpg")
                    cv2.imwrite(img_path, frame)
                    count += 1
                    last_save_time = current_time
                    print(f"✓ Saved {count}/{num_samples}: {img_path}")

                    # Visual feedback
                    feedback_frame = annotated_frame.copy()
                    cv2.rectangle(feedback_frame, (0, 0),
                                  (feedback_frame.shape[1],
                                   feedback_frame.shape[0]),
                                  (0, 255, 0), 10)
                    cv2.imshow("Data Collection - Static Gestures",
                               feedback_frame)
                    cv2.waitKey(100)

            elif key == ord('q'):
                print(f"\nCollection stopped by user")
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print(f"✓ Collection complete!")
        print(f"Total samples collected: {count}")
        print(f"Saved to: {class_dir}")
        print("="*60)

    def collect_dynamic_gestures(self, class_name, num_samples=50, duration=3):
        """
        Collect dynamic gesture data (video sequences)

        Args:
            class_name: Name of the gesture (e.g., 'hello', 'thanks')
            num_samples: Number of video sequences to collect
            duration: Duration of each video in seconds
        """
        class_dir = os.path.join(self.save_dir, 'dynamic', class_name)
        os.makedirs(class_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        count = 0
        recording = False
        frames_buffer = []

        print("\n" + "="*60)
        print(f"COLLECTING DYNAMIC GESTURE: '{class_name}'")
        print("="*60)
        print(f"Target: {num_samples} video samples")
        print(f"Duration: {duration} seconds per video")
        print("\nInstructions:")
        print("1. Press 'r' to start recording")
        print("2. Perform the gesture smoothly")
        print("3. Recording stops automatically after 3 seconds")
        print("4. Press 'q' to finish early")
        print("\nTips:")
        print("- Vary speed (fast, medium, slow)")
        print("- Vary position and orientation")
        print("- Keep hand visible throughout")
        print("\nStarting in 3 seconds...")
        time.sleep(3)

        recording_start_time = 0

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect hands
            hands_data, annotated_frame = self.detector.detect_hands(frame)

            # Handle recording
            if recording:
                elapsed = time.time() - recording_start_time
                remaining = duration - elapsed

                if remaining > 0:
                    frames_buffer.append(frame.copy())

                    # Recording indicator
                    cv2.circle(annotated_frame, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "RECORDING",
                                (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Time: {remaining:.1f}s",
                                (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    # Save video
                    recording = False
                    if len(frames_buffer) > 0:
                        video_path = os.path.join(
                            class_dir, f"{class_name}_{count:04d}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(
                            video_path, fourcc, fps, (frame_width, frame_height))

                        for f in frames_buffer:
                            out.write(f)

                        out.release()
                        count += 1
                        frames_buffer = []

                        print(f"✓ Saved {count}/{num_samples}: {video_path}")

            # Display interface
            if not recording:
                # Progress
                progress = (count / num_samples) * 100
                cv2.rectangle(annotated_frame, (0, 0),
                              (annotated_frame.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"Gesture: {class_name}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Collected: {count}/{num_samples} ({progress:.1f}%)",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Ready indicator
                if len(hands_data) > 0:
                    cv2.putText(annotated_frame, "Ready! Press 'r' to record",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "Show hand to start",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(annotated_frame, "Controls: 'r'=record | 'q'=quit",
                        (20, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Data Collection - Dynamic Gestures", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') and not recording and len(hands_data) > 0:
                recording = True
                recording_start_time = time.time()
                frames_buffer = []
                print(f"Recording {count + 1}/{num_samples}...")

            elif key == ord('q'):
                print(f"\nCollection stopped by user")
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print(f"✓ Collection complete!")
        print(f"Total samples collected: {count}")
        print(f"Saved to: {class_dir}")
        print("="*60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Collect sign language gesture data')
    parser.add_argument('--mode', type=str, choices=['static', 'dynamic'], required=True,
                        help='Collection mode: static (images) or dynamic (videos)')
    parser.add_argument('--gesture', type=str,
                        help='Gesture name (optional, will prompt if not provided)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to collect')
    parser.add_argument('--duration', type=int, default=3,
                        help='Duration of each video (for dynamic mode)')
    parser.add_argument('--save_dir', type=str,
                        default='data/raw', help='Directory to save data')

    args = parser.parse_args()

    collector = DataCollector(save_dir=args.save_dir)

    # Get gesture name if not provided
    gesture_name = args.gesture
    if not gesture_name:
        print("\n" + "="*60)
        print("DATA COLLECTION")
        print("="*60)
        gesture_name = input(
            f"\nEnter gesture name (e.g., A, B, hello, thanks): ").strip()

        if not gesture_name:
            print("✗ Error: Gesture name cannot be empty")
            return

    # Collect data
    if args.mode == 'static':
        collector.collect_static_gestures(gesture_name, args.samples)
    else:
        collector.collect_dynamic_gestures(
            gesture_name, args.samples, args.duration)

    print("\n✓ Data collection completed!")
    print("\nNext steps:")
    print("1. Collect more gestures by running this script again")
    print("2. When done, run: python src/utils/data_preparation.py")


if __name__ == "__main__":
    main()
