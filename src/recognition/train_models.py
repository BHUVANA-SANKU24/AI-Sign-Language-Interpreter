"""
Model Training Script
Train static or dynamic gesture recognition models
"""

from models.dynamic_gesture_lstm import DynamicGestureModel
from models.static_gesture_cnn import StaticGestureModel
import numpy as np
import sys
import os
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")

    try:
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    except:
        plt.close()


def train_static_model(data_dir='data/processed'):
    """Train static gesture recognition model"""
    print("\n" + "="*60)
    print("TRAINING STATIC GESTURE MODEL")
    print("="*60)

    # Check if data exists
    X_path = os.path.join(data_dir, 'X_static.npy')
    y_path = os.path.join(data_dir, 'y_static.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"✗ Data not found in {data_dir}")
        print("\nPlease prepare data first:")
        print(
            "  python src/utils/data_preparation.py --mode static --dataset data/raw/static")
        return None, None

    # Load data
    print("\nLoading data...")
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"✓ Loaded {len(X)} samples")
    print(f"  Feature dimensions: {X.shape[1]}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    num_classes = len(label_encoder.classes_)
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(label_encoder.classes_)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nData split:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Build model
    print("\nBuilding model...")
    input_shape = (X.shape[1],)
    model = StaticGestureModel(num_classes, input_shape)

    print("\nModel architecture:")
    model.summary()

    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("This may take 15-30 minutes depending on your hardware")
    print("="*60)

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )

    # Create models directory
    os.makedirs('data/models', exist_ok=True)

    # Save model
    model.save('data/models/static_gesture_model.h5')

    # Save label encoder
    np.save('data/models/static_label_encoder.npy', label_encoder.classes_)
    print(f"✓ Label encoder saved to: data/models/static_label_encoder.npy")

    # Plot history
    plot_training_history(history, 'data/models/static_training_history.png')

    # Final evaluation
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])

    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(
        f"Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(
        f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(
        f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"\nModel saved to: data/models/static_gesture_model.h5")
    print(f"{'='*60}")

    return model, label_encoder


def train_dynamic_model(data_dir='data/processed'):
    """Train dynamic gesture recognition model"""
    print("\n" + "="*60)
    print("TRAINING DYNAMIC GESTURE MODEL")
    print("="*60)

    # Check if data exists
    X_path = os.path.join(data_dir, 'X_dynamic.npy')
    y_path = os.path.join(data_dir, 'y_dynamic.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"✗ Data not found in {data_dir}")
        print("\nPlease prepare data first:")
        print("  python src/utils/data_preparation.py --mode dynamic --dataset data/raw/dynamic")
        return None, None

    # Load data
    print("\nLoading data...")
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"✓ Loaded {len(X)} sequences")
    print(f"  Sequence shape: {X.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    num_classes = len(label_encoder.classes_)
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(label_encoder.classes_)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nData split:")
    print(f"  Train sequences: {len(X_train)}")
    print(f"  Validation sequences: {len(X_val)}")

    # Build model
    print("\nBuilding model...")
    sequence_length = X.shape[1]
    feature_dim = X.shape[2]
    model = DynamicGestureModel(num_classes, sequence_length, feature_dim)

    print("\nModel architecture:")
    model.summary()

    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("This may take 30-60 minutes depending on your hardware")
    print("="*60)

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=16
    )

    # Create models directory
    os.makedirs('data/models', exist_ok=True)

    # Save model
    model.save('data/models/dynamic_gesture_model.h5')

    # Save label encoder
    np.save('data/models/dynamic_label_encoder.npy', label_encoder.classes_)
    print(f"✓ Label encoder saved to: data/models/dynamic_label_encoder.npy")

    # Plot history
    plot_training_history(history, 'data/models/dynamic_training_history.png')

    # Final evaluation
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])

    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(
        f"Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(
        f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(
        f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"\nModel saved to: data/models/dynamic_gesture_model.h5")
    print(f"{'='*60}")

    return model, label_encoder


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train gesture recognition models')
    parser.add_argument('--model', type=str, choices=['static', 'dynamic', 'both'],
                        default='static', help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')

    args = parser.parse_args()

    if args.model == 'static' or args.model == 'both':
        train_static_model(args.data_dir)

    if args.model == 'dynamic' or args.model == 'both':
        train_dynamic_model(args.data_dir)

    print("\n" + "="*60)
    print("✓ ALL TRAINING COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the model: python tests/test_static_recognition.py")
    print("2. Evaluate performance: python tests/evaluate_model.py")
    print("3. Run the full app: python src/app.py")
    print("="*60)


if __name__ == "__main__":
    main()
