import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.sign_animator import SignAnimator
from src.generation.nlp_processor import NLPProcessor

def test_generation():
    print("="*60)
    print("VERIFYING GENERATION MODE")
    print("="*60)
    
    try:
        nlp = NLPProcessor()
        animator = SignAnimator()
        
        test_text = "HELLO WORLD"
        print(f"\nProcessing text: '{test_text}'")
        
        sign_vocabulary = animator.list_available_signs()
        sign_sequence = nlp.text_to_sign_sequence(test_text, sign_vocabulary)
        
        print(f"Sign sequence: {' → '.join(sign_sequence)}")
        
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "test_generation_output.mp4")
        
        print(f"\nGenerating video: {output_path}...")
        result = animator.generate_sign_video(sign_sequence, output_path, fps=30)
        
        if result and os.path.exists(result):
            size = os.path.getsize(result)
            print(f"✓ Success! Video generated: {result} ({size} bytes)")
            return True
        else:
            print("✗ Failed to generate video")
            return False
            
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)
