"""
Static Gesture Recognition Model
CNN-based classifier for static hand gestures (letters, numbers)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class StaticGestureModel:
    """CNN model for static gesture recognition"""

    def __init__(self, num_classes, input_shape):
        """
        Initialize model

        Args:
            num_classes: Number of gesture classes
            input_shape: Shape of input features (feature_dim,)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
        print(f"✓ Static gesture model created")
        print(f"  Input shape: {input_shape}")
        print(f"  Output classes: {num_classes}")

    def _build_model(self):
        """Build CNN model architecture"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Dense layers for feature processing
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
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
              epochs=50, batch_size=32, callbacks=None):
        """
        Train the model

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: Optional Keras callbacks

        Returns:
            history: Training history
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    'data/models/best_static_model.h5',
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

    def predict(self, features):
        """Predict gesture from features"""
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)

        predictions = self.model.predict(features, verbose=0)
        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return results

    def save(self, path):
        """Save model to file"""
        self.model.save(path)
        print(f"✓ Model saved to: {path}")

    def load(self, path):
        """Load model from file"""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from: {path}")

    def summary(self):
        """Print model summary"""
        self.model.summary()
