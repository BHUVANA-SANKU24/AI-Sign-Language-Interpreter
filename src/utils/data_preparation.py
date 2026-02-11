"""
Data Preparation Module
Process collected images/videos and extract hand landmarks
"""

from src.recognition.hand_detection import HandDetector
import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DatasetPreparator:
    """Prepare dataset for training"""

    def __init__(self, detector=None):
        """Initialize with hand detector"""
        self.detector = detector if detector else HandDetector()
        print("✓ Dataset preparator initialized")

    def prepare_static_dataset(self, dataset_path, output_path='data/processed'):
        """
        Process static gesture dataset (images)

        Args:
            dataset_path: Path to raw image dataset
            output_path: Path to save processed data
        """
        print("="*60)
        print("PROCESSING STATIC DATASET")
        print("="*60)
        print(f"Dataset path: {dataset_path}")
        print(f"Output path: {output_path}")

        X_data = []
        y_labels = []
        class_names = []

        # Get all class directories
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset path does not exist: {dataset_path}")
            return None, None, None

        class_dirs = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]

        if not class_dirs:
            print(f"✗ No class directories found in {dataset_path}")
            return None, None, None

        print(f"\nFound {len(class_dirs)} classes: {class_dirs}")

        for class_name in sorted(class_dirs):
            print(f"\n{'─'*60}")
            print(f"Processing class: {class_name}")
            print(f"{'─'*60}")

            class_path = os.path.join(dataset_path, class_name)

            # Get all image files
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"Found {len(image_files)} images")

            processed_count = 0
            skipped_count = 0

            for img_file in tqdm(image_files, desc=f"  {class_name}"):
                img_path = os.path.join(class_path, img_file)

                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    skipped_count += 1
                    continue

                # Detect hand and extract features
                hands_data, _ = self.detector.detect_hands(image)

                if len(hands_data) > 0:
                    # Extract features from first detected hand
                    features = self.detector.extract_features(
                        hands_data[0]['landmarks'])
                    X_data.append(features)
                    y_labels.append(class_name)
                    processed_count += 1
                else:
                    skipped_count += 1

            print(f"  ✓ Processed: {processed_count}/{len(image_files)}")
            if skipped_count > 0:
                print(f"  ⚠ Skipped: {skipped_count} (no hand detected)")

            class_names.append(class_name)

        if not X_data:
            print("\n✗ No data processed! Check your dataset.")
            return None, None, None

        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_labels)

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save processed data
        np.save(os.path.join(output_path, 'X_static.npy'), X)
        np.save(os.path.join(output_path, 'y_static.npy'), y)

        # Save class names
        with open(os.path.join(output_path, 'classes_static.txt'), 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

        print(f"\n{'='*60}")
        print("✓ STATIC DATASET PREPARED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Feature dimensions: {X.shape[1]}")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}")

        return X, y, class_names

    def prepare_dynamic_dataset(self, dataset_path, output_path='data/processed',
                                sequence_length=30):
        """
        Process dynamic gesture dataset (videos)

        Args:
            dataset_path: Path to raw video dataset
            output_path: Path to save processed data
            sequence_length: Fixed length for all sequences
        """
        print("="*60)
        print("PROCESSING DYNAMIC DATASET")
        print("="*60)
        print(f"Dataset path: {dataset_path}")
        print(f"Output path: {output_path}")
        print(f"Sequence length: {sequence_length} frames")

        X_sequences = []
        y_labels = []
        class_names = []

        # Get all class directories
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset path does not exist: {dataset_path}")
            return None, None, None

        class_dirs = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]

        if not class_dirs:
            print(f"✗ No class directories found in {dataset_path}")
            return None, None, None

        print(f"\nFound {len(class_dirs)} classes: {class_dirs}")

        for class_name in sorted(class_dirs):
            print(f"\n{'─'*60}")
            print(f"Processing class: {class_name}")
            print(f"{'─'*60}")

            class_path = os.path.join(dataset_path, class_name)

            # Get all video files
            video_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.mp4', '.avi', '.mov'))]

            print(f"Found {len(video_files)} videos")

            processed_count = 0
            skipped_count = 0

            for video_file in tqdm(video_files, desc=f"  {class_name}"):
                video_path = os.path.join(class_path, video_file)

                # Extract sequence from video
                sequence = self._extract_sequence_from_video(
                    video_path, sequence_length)

                if sequence is not None:
                    X_sequences.append(sequence)
                    y_labels.append(class_name)
                    processed_count += 1
                else:
                    skipped_count += 1

            print(f"  ✓ Processed: {processed_count}/{len(video_files)}")
            if skipped_count > 0:
                print(f"  ⚠ Skipped: {skipped_count} (insufficient frames)")

            class_names.append(class_name)

        if not X_sequences:
            print("\n✗ No data processed! Check your dataset.")
            return None, None, None

        # Convert to numpy arrays
        X = np.array(X_sequences)
        y = np.array(y_labels)

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save processed data
        np.save(os.path.join(output_path, 'X_dynamic.npy'), X)
        np.save(os.path.join(output_path, 'y_dynamic.npy'), y)

        # Save class names
        with open(os.path.join(output_path, 'classes_dynamic.txt'), 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

        print(f"\n{'='*60}")
        print("✓ DYNAMIC DATASET PREPARED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total sequences: {len(X)}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}")

        return X, y, class_names

    def _extract_sequence_from_video(self, video_path, max_frames=30):
        """Extract landmark sequence from video"""
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []

        while len(landmarks_sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            hands_data, _ = self.detector.detect_hands(frame)

            if len(hands_data) > 0:
                features = self.detector.extract_features(
                    hands_data[0]['landmarks'])
                landmarks_sequence.append(features)

        cap.release()

        # Check if we got any frames
        if len(landmarks_sequence) == 0:
            return None

        # Convert to numpy array
        sequence = np.array(landmarks_sequence)

        # Pad or truncate to fixed length
        if len(sequence) < max_frames:
            # Pad with zeros
            padding = np.zeros((max_frames - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        else:
            # Truncate
            sequence = sequence[:max_frames]

        return sequence


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Prepare sign language dataset')
    parser.add_argument('--mode', type=str, choices=['static', 'dynamic', 'both'],
                        default='static', help='Processing mode')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to raw dataset')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Sequence length for dynamic gestures')

    args = parser.parse_args()

    preparator = DatasetPreparator()

    if args.mode == 'static' or args.mode == 'both':
        preparator.prepare_static_dataset(args.dataset, args.output)

    if args.mode == 'dynamic' or args.mode == 'both':
        preparator.prepare_dynamic_dataset(
            args.dataset, args.output, args.sequence_length
        )

    print("\n✓ Data preparation completed!")
    print("\nNext steps:")
    print("1. Run: python src/recognition/train_models.py --model static")
    print("2. Or: python src/recognition/train_models.py --model dynamic")


if __name__ == "__main__":
    main()
