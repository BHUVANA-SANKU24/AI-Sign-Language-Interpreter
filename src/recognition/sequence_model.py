"""
Sequence Detection Module
Detect gesture sequences and build words/sentences
"""

import numpy as np
from collections import deque
import time


class SequenceDetector:
    """Detect gesture sequences over time"""

    def __init__(self, recognizer, sequence_length=30):
        """
        Initialize sequence detector

        Args:
            recognizer: GestureRecognizer instance
            sequence_length: Length of sequence buffer
        """
        self.recognizer = recognizer
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        self.last_gesture = None
        self.gesture_stable_count = 0
        self.stability_threshold = 5  # Frames needed for stable detection

    def add_frame(self, features):
        """Add frame features to buffer"""
        self.frame_buffer.append(features)

    def detect_stable_gesture(self, features):
        """
        Detect stable gesture (reduces flickering)

        Args:
            features: Current frame features

        Returns:
            gesture: Stable gesture or None
            confidence: Confidence score
        """
        gesture, confidence = self.recognizer.recognize(features)

        if gesture == self.last_gesture:
            self.gesture_stable_count += 1
        else:
            self.gesture_stable_count = 1
            self.last_gesture = gesture

        # Only return gesture if it's been stable
        if self.gesture_stable_count >= self.stability_threshold:
            return gesture, confidence

        return None, confidence

    def reset(self):
        """Reset buffer and state"""
        self.frame_buffer.clear()
        self.last_gesture = None
        self.gesture_stable_count = 0


class WordBuilder:
    """Build words and sentences from detected gestures"""

    def __init__(self, space_threshold=2.0):
        """
        Initialize word builder

        Args:
            space_threshold: Seconds of no gesture to insert space
        """
        self.current_word = []
        self.completed_words = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.space_threshold = space_threshold

    def add_gesture(self, gesture, timestamp=None):
        """
        Add detected gesture

        Args:
            gesture: Detected gesture letter/sign
            timestamp: Current timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = time.time()

        # Check for word boundary (pause)
        if self.last_gesture_time > 0:
            time_diff = timestamp - self.last_gesture_time
            if time_diff > self.space_threshold:
                self._complete_word()

        # Add gesture if different from last
        if gesture and gesture != self.last_gesture:
            self.current_word.append(gesture)
            self.last_gesture = gesture
            self.last_gesture_time = timestamp

    def _complete_word(self):
        """Complete current word and add to sentence"""
        if self.current_word:
            word = ''.join(self.current_word)
            self.completed_words.append(word)
            self.current_word = []

    def get_current_text(self):
        """Get current text including incomplete word"""
        text_parts = self.completed_words.copy()
        if self.current_word:
            text_parts.append(''.join(self.current_word))
        return ' '.join(text_parts)

    def get_sentence(self):
        """Get completed sentence only"""
        return ' '.join(self.completed_words)

    def finalize(self):
        """Finalize current word and return complete sentence"""
        self._complete_word()
        sentence = self.get_sentence()
        return sentence

    def clear(self):
        """Clear all text"""
        self.current_word = []
        self.completed_words = []
        self.last_gesture = None
        self.last_gesture_time = 0
