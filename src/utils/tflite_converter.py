"""
Convert trained Keras models to TFLite format for optimization
"""
import tensorflow as tf
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def convert_to_tflite(h5_model_path, tflite_output_path):
    if not os.path.exists(h5_model_path):
        print(f"⚠ Model not found: {h5_model_path}")
        return False
        
    print(f"Converting {h5_model_path} to TFLite...")
    model = tf.keras.models.load_model(h5_model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # To reduce size and potentially boost speed while maintaining accuracy
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"✓ Saved TFLite model to {tflite_output_path}")
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TFLITE CONVERSION PROCESS")
    print("="*60)
    
    os.makedirs('data/models', exist_ok=True)
    
    # Convert static and dynamic models if they exist
    convert_to_tflite('data/models/static_gesture_model.h5', 'data/models/static_gesture_model.tflite')
    convert_to_tflite('data/models/dynamic_gesture_model.h5', 'data/models/dynamic_gesture_model.tflite')
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
