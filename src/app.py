"""
Sign Language Interpreter - Main Application
Two-way communication system for sign language
"""

from src.generation.sign_animator import SignAnimator
from src.generation.nlp_processor import NLPProcessor
from src.generation.speech_recognition import SpeechToText
from src.recognition.text_to_speech import TextToSpeech
from src.recognition.text_correction import TextCorrector
from src.recognition.sequence_model import SequenceDetector, WordBuilder
from src.recognition.gesture_recognition import GestureRecognizer
from src.recognition.hand_detection import HandDetector
import cv2
import time
import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SignLanguageInterpreter:
    """Main application class"""

    def __init__(self):
        print("\n" + "="*60)
        print("SIGN LANGUAGE INTERPRETER")
        print("Two-Way Communication System")
        print("="*60)

        # Initialize components
        print("\nInitializing components...")

        self.hand_detector = HandDetector()
        self.tts = TextToSpeech()
        self.stt = SpeechToText()
        self.nlp = NLPProcessor()
        self.animator = SignAnimator()
        self.text_corrector = TextCorrector(enable_spell_check=True,
                                            enable_grammar_check=False)

        # Load model
        self.recognizer = None
        self.sequence_detector = None
        self._load_recognition_model()

        self.mode = "recognition"
        self.running = False

    def _load_recognition_model(self):
        """Load trained recognition model"""
        model_path = 'data/models/static_gesture_model.h5'
        encoder_path = 'data/models/static_label_encoder.npy'

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                self.recognizer = GestureRecognizer(
                    model_path, encoder_path, confidence_threshold=0.7
                )
                self.sequence_detector = SequenceDetector(self.recognizer)
                print("✓ Recognition model loaded")
                return True
            except Exception as e:
                print(f"⚠ Could not load model: {e}")
                return False
        else:
            print("⚠ No trained model found")
            print("  Train model: python src/recognition/train_models.py --model static")
            return False

    def recognition_mode(self):
        """Recognition Mode: Sign Language → Text → Speech"""
        if self.recognizer is None:
            print("\n✗ No model loaded. Cannot run recognition mode.")
            print("Train a model first: python src/recognition/train_models.py")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return

        print("\n" + "="*60)
        print("RECOGNITION MODE ACTIVE")
        print("="*60)
        print("Controls:")
        print("  's' - Switch to generation mode")
        print("  'c' - Clear and speak current text")
        print("  'r' - Reset/clear text")
        print("  'q' - Quit")
        print("="*60)
        print("\nStarting...")

        word_builder = WordBuilder(space_threshold=2.0)
        last_spoken = ""

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect hands
            hands_data, annotated_frame = self.hand_detector.detect_hands(
                frame)

            # Create UI overlay
            self._draw_recognition_ui(annotated_frame)

            # Predict gesture
            if len(hands_data) > 0:
                features = self.hand_detector.extract_features(
                    hands_data[0]['landmarks'])
                gesture, confidence = self.sequence_detector.detect_stable_gesture(
                    features)

                if gesture:
                    word_builder.add_gesture(gesture, time.time())

                    # Display current gesture
                    cv2.putText(annotated_frame, f"{gesture} ({confidence:.0%})",
                                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Get and display current text
            current_text = word_builder.get_current_text()
            self._display_text(annotated_frame, current_text)

            # Show frame
            cv2.imshow("Sign Language Interpreter", annotated_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.mode = "generation"
                break
            elif key == ord('c'):
                # Speak current text
                if current_text and current_text != last_spoken:
                    corrected = self.text_corrector.correct_text(current_text)
                    print(f"\nOriginal: {current_text}")
                    print(f"Corrected: {corrected}")
                    self.tts.speak(corrected)
                    last_spoken = current_text
            elif key == ord('r'):
                # Reset text
                word_builder.clear()
                last_spoken = ""

        cap.release()
        cv2.destroyAllWindows()

    def generation_mode(self):
        """Generation Mode: Speech/Text → Sign Language Animation"""
        print("\n" + "="*60)
        print("GENERATION MODE ACTIVE")
        print("="*60)
        print("Options:")
        print("  1. Record speech (r)")
        print("  2. Type text (t)")
        print("  3. Switch to recognition (s)")
        print("  4. Quit (q)")
        print("="*60)

        while self.running:
            choice = input("\nChoose option: ").strip().lower()

            if choice == 'q':
                self.running = False
                break
            elif choice == 's':
                self.mode = "recognition"
                break
            elif choice == 'r':
                if self.stt.is_available():
                    text = self.stt.listen(timeout=5, phrase_time_limit=10)
                    if text:
                        self._generate_signs(text)
                else:
                    print("⚠ Speech recognition not available")
            elif choice == 't':
                text = input("\nEnter text: ").strip()
                if text:
                    self._generate_signs(text)
            else:
                print("Invalid option. Try again.")

    def _generate_signs(self, text):
        """Generate and display sign animation"""
        print(f"\nGenerating signs for: '{text}'")

        # Process text
        sign_vocabulary = self.animator.list_available_signs()
        sign_sequence = self.nlp.text_to_sign_sequence(text, sign_vocabulary)

        print(f"Sign sequence: {' → '.join(sign_sequence)}")

        # Generate video
        timestamp = int(time.time())
        output_path = f"output/sign_animation_{timestamp}.mp4"

        result = self.animator.generate_sign_video(
            sign_sequence, output_path, fps=30)

        if result:
            self._play_video(result)
        else:
            print("✗ Failed to generate animation")

    def _play_video(self, video_path):
        """Play generated sign video"""
        print(f"\nPlaying: {video_path}")
        print("Press 'q' to stop, 'r' to replay, space to pause")

        cap = cv2.VideoCapture(video_path)
        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                cv2.imshow("Sign Language Animation", frame)

            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False

        cap.release()
        cv2.destroyAllWindows()

    def _draw_recognition_ui(self, frame):
        """Draw UI overlay for recognition mode"""
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 50), (50, 50, 50), -1)
        cv2.putText(frame, "Mode: RECOGNITION",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Controls
        cv2.putText(frame, "s=Switch | c=Speak | r=Reset | q=Quit",
                    (w-400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _display_text(self, frame, text):
        """Display recognized text on frame"""
        h, w = frame.shape[:2]

        # Text box
        box_height = 100
        cv2.rectangle(frame, (0, h-box_height), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h-box_height), (w, h), (255, 255, 255), 2)

        # Display text (wrap if too long)
        max_width = w - 40
        font_scale = 0.7
        font = cv2.FONT_HERSHEY_SIMPLEX

        if len(text) > 60:
            text = text[-60:]  # Show last 60 characters

        cv2.putText(frame, text if text else "(gesture here...)",
                    (20, h-50), font, font_scale,
                    (255, 255, 255) if text else (100, 100, 100), 2)

    def run(self):
        """Main application loop"""
        print("\n✓ Initialization complete!")
        print(f"\nStarting in {self.mode.upper()} mode...")

        self.running = True

        try:
            while self.running:
                if self.mode == "recognition":
                    self.recognition_mode()
                elif self.mode == "generation":
                    self.generation_mode()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            cv2.destroyAllWindows()

        print("\n" + "="*60)
        print("Application closed. Thank you!")
        print("="*60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Sign Language Interpreter')
    parser.add_argument('--mode', type=str, choices=['recognition', 'generation'],
                        default='recognition', help='Starting mode')
    args = parser.parse_args()

    app = SignLanguageInterpreter()
    app.mode = args.mode
    app.run()


if __name__ == "__main__":
    main()
