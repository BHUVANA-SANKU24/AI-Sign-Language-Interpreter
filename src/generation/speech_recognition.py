"""
Speech Recognition Module
Convert speech input to text using microphone
"""

import speech_recognition as sr


class SpeechToText:
    """Convert speech to text"""

    def __init__(self):
        """Initialize speech recognizer"""
        self.recognizer = sr.Recognizer()
        self.available = False

        # Check if microphone is available
        try:
            self.microphone = sr.Microphone()

            # Adjust for ambient noise
            print("Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            self.available = True
            print("✓ Speech recognition initialized")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize microphone: {e}")
            print("  Speech input will not be available")
            self.microphone = None

    def listen(self, timeout=5, phrase_time_limit=10):
        """
        Listen to microphone and convert to text

        Args:
            timeout: Seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for phrase

        Returns:
            text: Recognized text or None
        """
        if not self.available:
            print("✗ Microphone not available")
            return None

        try:
            with self.microphone as source:
                print("🎤 Listening... Speak now!")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"✓ Recognized: {text}")
                return text

        except sr.WaitTimeoutError:
            print("⏱ No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"✗ Error with speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    def is_available(self):
        """Check if speech recognition is available"""
        return self.available
