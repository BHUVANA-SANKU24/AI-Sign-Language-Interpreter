"""
Sign Animation Module
Generate visual sign language from text
"""

import cv2
import numpy as np
import os
from pathlib import Path


class SignAnimator:
    """Generate sign language animations"""

    def __init__(self, sign_library_dir='data/sign_library'):
        """
        Initialize animator

        Args:
            sign_library_dir: Directory containing sign videos/images
        """
        self.sign_library_dir = sign_library_dir
        self.sign_library = self._load_sign_library()
        print(
            f"✓ Sign animator initialized: {len(self.sign_library)} signs loaded")

    def _load_sign_library(self):
        """Load available signs from library"""
        sign_library = {}

        if not os.path.exists(self.sign_library_dir):
            print(f"⚠ Sign library not found at {self.sign_library_dir}")
            print(f"  Creating directory...")
            os.makedirs(self.sign_library_dir, exist_ok=True)
            return sign_library

        # Load all video/image files
        for filename in os.listdir(self.sign_library_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.png', '.gif')):
                sign_name = os.path.splitext(filename)[0].upper()
                file_path = os.path.join(self.sign_library_dir, filename)
                sign_library[sign_name] = file_path

        return sign_library

    def generate_sign_video(self, sign_sequence, output_path, fps=30):
        """
        Generate video of sign sequence

        Args:
            sign_sequence: List of signs to perform
            output_path: Path to save output video
            fps: Frames per second

        Returns:
            output_path: Path to generated video
        """
        print(f"Generating animation for: {' → '.join(sign_sequence)}")

        frames = []

        for sign in sign_sequence:
            sign_frames = self._get_sign_frames(sign)
            frames.extend(sign_frames)

            # Add transition frames
            if len(sign_sequence) > 1:
                transition = self._create_transition_frames(3)
                frames.extend(transition)

        # Create video from frames
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)

            out.release()
            print(f"✓ Animation saved: {output_path} ({len(frames)} frames)")
            return output_path
        else:
            print("✗ No frames generated")
            return None

    def _get_sign_frames(self, sign):
        """Get frames for a specific sign"""
        sign_upper = sign.upper()

        if sign_upper in self.sign_library:
            file_path = self.sign_library[sign_upper]

            # Check if video or image
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                return self._load_video_frames(file_path)
            else:
                return self._load_image_as_frames(file_path, duration_frames=30)
        else:
            # Generate placeholder
            return self._generate_placeholder(sign, num_frames=30)

    def _load_video_frames(self, video_path):
        """Load frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _load_image_as_frames(self, image_path, duration_frames=30):
        """Load image and display for duration"""
        image = cv2.imread(image_path)
        if image is None:
            return self._generate_placeholder("?", num_frames=duration_frames)

        return [image.copy() for _ in range(duration_frames)]

    def _generate_placeholder(self, sign, num_frames=30):
        """Generate placeholder animation for missing sign"""
        frames = []

        for i in range(num_frames):
            # Create white background
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

            # Add blue header
            cv2.rectangle(frame, (0, 0), (640, 80), (200, 100, 50), -1)
            cv2.putText(frame, "Sign Language",
                        (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add sign text with animation
            scale = 1.0 + 0.1 * np.sin(i * np.pi / 15)
            text_size = cv2.getTextSize(sign, cv2.FONT_HERSHEY_BOLD, 2, 3)[0]
            text_x = (640 - int(text_size[0] * scale)) // 2
            text_y = 280

            cv2.putText(frame, sign,
                        (text_x, text_y), cv2.FONT_HERSHEY_BOLD, 2 * scale, (0, 0, 0), 3)

            # Add note
            cv2.putText(frame, "(Placeholder - Add video to data/sign_library/)",
                        (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            frames.append(frame)

        return frames

    def _create_transition_frames(self, num_frames=3):
        """Create transition frames between signs"""
        frames = []

        for i in range(num_frames):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            frames.append(frame)

        return frames

    def has_sign(self, sign):
        """Check if sign is available in library"""
        return sign.upper() in self.sign_library

    def list_available_signs(self):
        """List all available signs"""
        return list(self.sign_library.keys())
