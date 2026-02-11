"""
Gesture Recognition Module
Real-time gesture recognition using trained models
"""

import numpy as np
import tensorflow as tf
from collections import deque
import time


class GestureRecognizer:
    """Real-time gesture recognition"""

    def __init__(self, model_path, label_encoder_path, confidence_threshold=0.7):
        """
        Initialize gesture recognizer

        Args:
            model_path: Path to trained model
            label_encoder_path: Path to label encoder
            confidence_threshold: Minimum confidence for prediction
        """
        self.model = tf.keras.models.load_model(model_path)
        self.classes = np.load(label_encoder_path, allow_pickle=True)
        self.confidence_threshold = confidence_threshold

        print(f"✓ Model loaded: {len(self.classes)} gestures")
        print(f"  Classes: {list(self.classes)}")

    def recognize(self, features):
        """
        Recognize gesture from features

        Args:
            features: Extracted hand features

        Returns:
            gesture: Recognized gesture
            confidence: Prediction confidence
        """
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)

        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)

        if confidence >= self.confidence_threshold:
            gesture = self.classes[predicted_class]
            return gesture, confidence

        return None, confidence

    def get_top_predictions(self, features, top_k=3):
        """Get top K predictions"""
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)

        predictions = self.model.predict(features, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'gesture': self.classes[idx],
                'confidence': predictions[idx]
            })

        return results
