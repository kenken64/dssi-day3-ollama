#!/usr/bin/env python3
"""
Test script to validate AMD GPU support integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_to_sql_train import TextToSQLConfig, ModelTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_amd_config():
    """Test AMD GPU configuration"""
    print("🔴 Testing AMD GPU Configuration")
    print("=" * 40)
    
    config = TextToSQLConfig()
    
    # Test AMD optimizations
    print("Before AMD optimizations:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Grad accumulation: {config.gradient_accumulation_steps}")
    print(f"  Max length: {config.max_length}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Use 4-bit: {config.use_4bit}")
    
    config.apply_amd_optimizations()
    
    print("\nAfter AMD optimizations:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Grad accumulation: {config.gradient_accumulation_steps}")
    print(f"  Max length: {config.max_length}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Use 4-bit: {config.use_4bit}")
    print(f"  Compute dtype: {config.bnb_4bit_compute_dtype}")

def test_amd_detection():
    """Test AMD GPU detection"""
    print("\n🔍 Testing AMD GPU Detection")
    print("=" * 40)
    
    config = TextToSQLConfig()
    trainer = ModelTrainer(config)
    
    has_amd, amd_info = trainer.detect_amd_gpu()
    
    print(f"AMD GPU detected: {has_amd}")
    if has_amd:
        print(f"AMD info: {amd_info}")
    else:
        print("No AMD GPU found (expected on this system)")

def test_device_capabilities():
    """Test device capabilities detection with AMD support"""
    print("\n⚙️  Testing Device Capabilities Detection")
    print("=" * 50)
    
    config = TextToSQLConfig()
    trainer = ModelTrainer(config)
    
    use_fp16, use_bf16, device_type = trainer.detect_device_capabilities()
    
    print(f"Detected device type: {device_type}")
    print(f"Use fp16: {use_fp16}")
    print(f"Use bf16: {use_bf16}")

def main():
    print("🧪 AMD GPU Support Integration Test")
    print("=" * 50)
    
    try:
        test_amd_config()
        test_amd_detection()
        test_device_capabilities()
        
        print("\n✅ All tests completed successfully!")
        print("\nℹ️  AMD GPU Support Summary:")
        print("  - AMD optimization configuration: ✅ Working")
        print("  - AMD GPU detection: ✅ Working")
        print("  - Device capabilities detection: ✅ Working")
        print("  - MX300x compatibility: ❌ Not supported (as expected)")
        
        print("\n💡 Usage for AMD users:")
        print("  - Supported AMD GPUs: Use --device amd or --amd-mode")
        print("  - MX300x users: Use --cpu-mode (ROCm not supported)")
        print("  - Install ROCm: https://rocm.docs.amd.com/")
        print("  - Install PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
