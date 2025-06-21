#!/usr/bin/env python3
"""
Device capability detection script for Text-to-SQL training
Shows what devices are available and recommends optimal settings
"""

import torch
import sys
import os

def check_cuda():
    """Check CUDA availability and capabilities"""
    print("🔍 CUDA GPU Detection:")
    print("-" * 30)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA Available: {device_count} GPU(s) detected")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            major, minor = torch.cuda.get_device_capability(i)
            
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f}GB")
            print(f"    Compute Capability: {major}.{minor}")
            
            # Recommend precision based on capability
            if major >= 8:
                print(f"    💡 Recommendation: Use bf16 for optimal training")
                print(f"    💡 Suggested command: --device cuda")
            elif major >= 7:
                print(f"    💡 Recommendation: Use fp16 for faster training")
                print(f"    💡 Suggested command: --device cuda")
            else:
                print(f"    💡 Recommendation: Use fp32 for compatibility")
                print(f"    💡 Suggested command: --device cuda --no-quantization")
        
        return True
    else:
        print("❌ CUDA Not Available")
        return False

def check_mps():
    """Check Apple Silicon MPS availability"""
    print("\n🍎 Apple Silicon MPS Detection:")
    print("-" * 35)
    
    if torch.backends.mps.is_available():
        print("✅ MPS Available (Apple Silicon Mac)")
        print("  💡 Recommendation: Use MPS with conservative settings")
        print("  💡 Suggested command: --device mps")
        print("  💡 For fastest training: --fast-mac")
        
        # Test MPS basic functionality
        try:
            test_tensor = torch.randn(1, 1, device='mps')
            print("  ✅ MPS basic test passed")
            del test_tensor
        except Exception as e:
            print(f"  ⚠️  MPS test failed: {e}")
        
        return True
    else:
        print("❌ MPS Not Available (Intel Mac or non-Mac system)")
        return False

def check_cpu():
    """Check CPU capabilities"""
    print("\n🖥️  CPU Detection:")
    print("-" * 20)
    
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        print(f"✅ CPU: {cpu_count} cores")
        print(f"✅ RAM: {memory_gb:.1f}GB")
        
        if memory_gb >= 16:
            print("  💡 Recommendation: CPU training possible with smaller models")
            print("  💡 Suggested command: --cpu-mode --max-samples 500")
        elif memory_gb >= 8:
            print("  💡 Recommendation: CPU training with very small dataset")
            print("  💡 Suggested command: --cpu-mode --max-samples 250")
        else:
            print("  ⚠️  Low memory - CPU training may be slow")
            print("  💡 Suggested command: --cpu-mode --max-samples 100")
            
    except ImportError:
        print("📦 Install psutil for detailed CPU info: pip install psutil")
    
    return True

def recommend_settings():
    """Provide overall recommendations"""
    print("\n🎯 Training Recommendations:")
    print("-" * 30)
    
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    
    if has_cuda:
        print("🏆 BEST: Use NVIDIA GPU for fastest training")
        print("   Command: python text_to_sql_train.py --device cuda")
        
        # Check if high-end GPU
        if has_cuda:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                print("   🚀 High-end GPU detected - can handle large datasets")
                print("   Command: python text_to_sql_train.py --device cuda --max-samples 10000")
            
    elif has_mps:
        print("🥈 GOOD: Use Apple Silicon MPS for good performance")
        print("   Command: python text_to_sql_train.py --device mps")
        print("   Fast mode: python text_to_sql_train.py --fast-mac")
        
    else:
        print("🥉 FALLBACK: Use CPU for compatibility")
        print("   Command: python text_to_sql_train.py --cpu-mode")
        print("   💡 Consider using a smaller model: --base-model Salesforce/codet5p-220m")

def show_example_commands():
    """Show example training commands for different scenarios"""
    print("\n📝 Example Training Commands:")
    print("-" * 35)
    
    examples = [
        {
            "name": "NVIDIA GPU (High-end)",
            "command": "python text_to_sql_train.py --device cuda --max-samples 10000",
            "description": "Full training with GPU acceleration"
        },
        {
            "name": "NVIDIA GPU (Basic)",
            "command": "python text_to_sql_train.py --device cuda --max-samples 5000",
            "description": "Balanced training with GPU"
        },
        {
            "name": "Apple Silicon Mac",
            "command": "python text_to_sql_train.py --device mps --max-samples 2000",
            "description": "MPS-optimized training"
        },
        {
            "name": "Mac Fast Mode",
            "command": "python text_to_sql_train.py --fast-mac",
            "description": "Fastest Mac training (for testing)"
        },
        {
            "name": "CPU Training",
            "command": "python text_to_sql_train.py --cpu-mode --max-samples 500",
            "description": "CPU-only training"
        },
        {
            "name": "Small Model (Any Device)",
            "command": "python text_to_sql_train.py --base-model Salesforce/codet5p-220m --max-samples 1000",
            "description": "Tiny model for quick testing"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}:")
        print(f"   {example['command']}")
        print(f"   📝 {example['description']}")
        print()

def main():
    print("🤖 Text-to-SQL Device Capability Checker")
    print("=" * 50)
    
    # Check all device types
    has_cuda = check_cuda()
    has_mps = check_mps()
    has_cpu = check_cpu()
    
    # Show recommendations
    recommend_settings()
    
    # Show example commands
    show_example_commands()
    
    print("🎉 Device check complete!")
    print("💡 Choose the command that matches your hardware capabilities.")

if __name__ == "__main__":
    main()
