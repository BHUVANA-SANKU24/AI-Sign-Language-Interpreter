"""
Text-to-Speech Module
Convert recognized text to spoken audio
"""

import pyttsx3


class TextToSpeech:
    """Convert text to speech"""

    def __init__(self, rate=150, volume=1.0):
        """
        Initialize TTS engine

        Args:
            rate: Speaking rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)

            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)

            self.available = True
            print("✓ Text-to-Speech initialized")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize TTS: {e}")
            self.engine = None
            self.available = False

    def speak(self, text):
        """
        Convert text to speech

        Args:
            text: Text to speak
        """
        if not self.available:
            print(f"TTS not available. Text: {text}")
            return

        if not text or text.strip() == "":
            return

        try:
            print(f"🔊 Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"✗ Error speaking text: {e}")

    def set_voice(self, voice_index=0):
        """Change voice"""
        if not self.available:
            return

        voices = self.engine.getProperty('voices')
        if voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)

    def set_rate(self, rate):
        """Change speaking rate"""
        if not self.available:
            return

        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """Change volume (0.0 to 1.0)"""
        if not self.available:
            return

        self.engine.setProperty('volume', volume)

    def list_voices(self):
        """List available voices"""
        if not self.available:
            return []

        voices = self.engine.getProperty('voices')
        voice_list = []
        for i, voice in enumerate(voices):
            voice_list.append({
                'index': i,
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages
            })
        return voice_list
