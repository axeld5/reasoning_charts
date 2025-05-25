#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Chart VQA

This script evaluates multiple vision-language models on chart VQA tasks and 
includes fine-tuning of SmolVLM2. It also explains why Gemma 3 1B cannot be used.

Models evaluated:
- SmolVLM2-256M-Video-Instruct (original + fine-tuned)
- Qwen-2.5-VL-3B-Instruct

Models NOT evaluated (and why):
- Gemma 3 1B: Text-only model, no vision capabilities
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required dependencies"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available. This will be very slow on CPU!")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    print("🚀 Starting Comprehensive Chart VQA Model Evaluation")
    print("="*60)
    
    # Check if running in correct directory
    if not os.path.exists("finetuning.py"):
        print("❌ Error: finetuning.py not found in current directory")
        print("Please run this script from the project root directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check GPU availability
    check_gpu()
    
    print("\n📊 About to evaluate the following models:")
    print("✅ SmolVLM2-256M-Video-Instruct (original)")
    print("✅ Qwen-2.5-VL-3B-Instruct") 
    print("✅ SmolVLM2-256M-Video-Instruct (fine-tuned)")
    print("❌ Gemma 3 1B (text-only, no vision support)")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("• Gemma 3 1B cannot process images - it's text-only")
    print("• Only Gemma 3 models with 4B+ parameters support vision")
    print("• This evaluation will take significant time and GPU memory")
    print("• Results will be saved to 'comprehensive_evaluation_results.json'")
    
    input("\n🔄 Press Enter to continue or Ctrl+C to cancel...")
    
    print("\n🏃 Running evaluation...")
    try:
        # Import and run the main script
        exec(open("finetuning.py").read())
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("Check the error details above for troubleshooting")

if __name__ == "__main__":
    main() 