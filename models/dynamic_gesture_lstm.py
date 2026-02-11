"""
Dynamic Gesture Recognition Model
LSTM-based classifier for dynamic hand gestures (words, phrases)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DynamicGestureModel:
    """LSTM model for dynamic gesture recognition"""

    def __init__(self, num_classes, sequence_length, feature_dim):
        """
        Initialize model

        Args:
            num_classes: Number of gesture classes
            sequence_length: Length of input sequences
            feature_dim: Dimension of features at each timestep
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = self._build_model()
        print(f"✓ Dynamic gesture model created")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Output classes: {num_classes}")

    def _build_model(self):
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.feature_dim)),

            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=16, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    'data/models/best_dynamic_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    verbose=1
                )
            ]

        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)

        return history

    def predict(self, sequence):
        """Predict gesture from sequence"""
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)

        predictions = self.model.predict(sequence, verbose=0)
        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return results

    def save(self, path):
        """Save model"""
        self.model.save(path)
        print(f"✓ Model saved to: {path}")

    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from: {path}")

    def summary(self):
        """Print model summary"""
        self.model.summary()
