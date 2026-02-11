"""
Model Evaluation
Generate detailed performance metrics and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_static_model():
    """Evaluate static gesture model"""
    print("="*60)
    print("STATIC MODEL EVALUATION")
    print("="*60)

    # Check if files exist
    if not os.path.exists('data/processed/X_static.npy'):
        print("✗ Data not found. Please prepare data first.")
        return

    if not os.path.exists('data/models/static_gesture_model.h5'):
        print("✗ Model not found. Please train model first.")
        return

    # Load data
    print("\nLoading data...")
    X = np.load('data/processed/X_static.npy')
    y = np.load('data/processed/y_static.npy')

    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model('data/models/static_gesture_model.h5')
    classes = np.load('data/models/static_label_encoder.npy',
                      allow_pickle=True)

    print(f"✓ Loaded {len(X)} samples, {len(classes)} classes")

    # Create label mapping
    label_to_int = {label: i for i, label in enumerate(classes)}
    y_encoded = np.array([label_to_int[label] for label in y])

    # Use last 20% as test set
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTest set: {len(X_test)} samples")

    # Predictions
    print("Making predictions...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # Classification report
    print("\nDetailed Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=classes))

    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(max(10, len(classes)), max(8, len(classes)-2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.2%}',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('data/models/confusion_matrix.png',
                dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved: data/models/confusion_matrix.png")

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(max(12, len(classes)*0.8), 6))
    bars = plt.bar(classes, class_accuracy, color='skyblue', edgecolor='navy')

    # Color bars based on accuracy
    for bar, acc in zip(bars, class_accuracy):
        if acc >= 0.9:
            bar.set_color('green')
        elif acc >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.axhline(y=accuracy, color='r', linestyle='--',
                label=f'Average: {accuracy:.2%}')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Gesture Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/models/class_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Class accuracy chart saved: data/models/class_accuracy.png")

    # Show plots
    try:
        plt.show(block=False)
        plt.pause(3)
        plt.close('all')
    except:
        plt.close('all')

    # Print worst performing classes
    print("\n" + "="*60)
    print("CLASS PERFORMANCE SUMMARY")
    print("="*60)
    worst_indices = np.argsort(class_accuracy)[:5]
    print("\nLowest accuracy classes:")
    for idx in worst_indices:
        print(f"  {classes[idx]}: {class_accuracy[idx]:.2%}")

    best_indices = np.argsort(class_accuracy)[-5:][::-1]
    print("\nHighest accuracy classes:")
    for idx in best_indices:
        print(f"  {classes[idx]}: {class_accuracy[idx]:.2%}")

    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  • data/models/confusion_matrix.png")
    print("  • data/models/class_accuracy.png")


if __name__ == "__main__":
    try:
        evaluate_static_model()
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
