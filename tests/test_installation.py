"""
Test Installation
Verifies that all required packages are installed correctly
"""

import sys


def test_imports():
    """Test all required imports"""
    print("="*60)
    print("TESTING INSTALLATION")
    print("="*60)

    all_ok = True

    # Test OpenCV
    print("\n1. Testing OpenCV...")
    try:
        import cv2
        print(f"   ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"   ✗ OpenCV failed: {e}")
        all_ok = False

    # Test MediaPipe
    print("\n2. Testing MediaPipe...")
    try:
        import mediapipe as mp
        print(f"   ✓ MediaPipe {mp.__version__}")
    except ImportError as e:
        print(f"   ✗ MediaPipe failed: {e}")
        all_ok = False

    # Test TensorFlow
    print("\n3. Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"   ✓ TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✓ GPU available: {len(gpus)} device(s)")
        else:
            print("   ℹ No GPU detected (will use CPU)")
    except ImportError as e:
        print(f"   ✗ TensorFlow failed: {e}")
        all_ok = False

    # Test NumPy
    print("\n4. Testing NumPy...")
    try:
        import numpy as np
        print(f"   ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"   ✗ NumPy failed: {e}")
        all_ok = False

    # Test other packages
    packages = [
        ('pandas', 'pd'),
        ('sklearn', None),
        ('nltk', None),
        ('pyttsx3', None),
        ('speech_recognition', 'sr'),
        ('matplotlib', 'plt'),
        ('PIL', None)
    ]

    print("\n5. Testing other packages...")
    for pkg_name, alias in packages:
        try:
            if alias:
                exec(f"import {pkg_name} as {alias}")
            else:
                exec(f"import {pkg_name}")
            print(f"   ✓ {pkg_name}")
        except ImportError:
            print(f"   ⚠ {pkg_name} (optional)")

    print("\n" + "="*60)
    if all_ok:
        print("✓ ALL CRITICAL PACKAGES INSTALLED!")
        print("="*60)
        print("\nYour environment is ready!")
        print("\nNext steps:")
        print("1. Test webcam: python tests/test_webcam.py")
        print("2. Test hand detection: python tests/test_hand_detection.py")
        return True
    else:
        print("✗ SOME PACKAGES FAILED")
        print("="*60)
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"\nPython Version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and 8 <= version.minor <= 11:
        print("✓ Python version is compatible")
        return True
    else:
        print("⚠ Recommended Python version: 3.8 - 3.11")
        print(f"  Your version: {version.major}.{version.minor}")
        return True  # Don't fail, just warn


if __name__ == "__main__":
    print("\nChecking Python version...")
    check_python_version()

    print("\nTesting package imports...")
    success = test_imports()

    sys.exit(0 if success else 1)
