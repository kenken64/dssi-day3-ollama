#!/usr/bin/env python3
"""
Device capability detection script for Text-to-SQL training
Shows what devices are available and recommends optimal settings
"""

import torch
import sys
import os
import subprocess

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

def check_amd():
    """Check AMD GPU (ROCm/HIP) availability and capabilities"""
    print("\n🔴 AMD GPU Detection:")
    print("-" * 25)
    
    # Check for AMD GPU environment variables
    hip_visible = os.environ.get('HIP_VISIBLE_DEVICES', None)
    rocm_path = os.environ.get('ROCM_PATH', None)
    
    # Try to detect AMD GPU through various methods
    amd_detected = False
    
    # Method 1: Check ROCm installation
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            amd_detected = True
            gpu_info = result.stdout.strip()
            print(f"✅ AMD GPU detected via ROCm: {gpu_info}")
        else:
            print("❌ ROCm not installed or no AMD GPU detected")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("❌ ROCm tools not available (rocm-smi not found)")
    
    # Method 2: Check if PyTorch ROCm is available
    try:
        # Check if PyTorch was compiled with ROCm support
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Check if this is actually ROCm (HIP) backend
            try:
                device_name = torch.cuda.get_device_name(0)
                if 'AMD' in device_name or 'Radeon' in device_name or 'gfx' in device_name:
                    amd_detected = True
                    print(f"✅ AMD GPU detected via PyTorch: {device_name}")
            except:
                pass
    except:
        pass
    
    # Method 3: Check system information for AMD GPU
    try:
        if sys.platform.startswith('linux'):
            result = subprocess.run(['lspci', '-nn', '|', 'grep', '-E', '"VGA|3D"'], 
                                  shell=True, capture_output=True, text=True, timeout=5)
            if 'AMD' in result.stdout or 'ATI' in result.stdout:
                amd_detected = True
                print("✅ AMD GPU detected in system (via lspci)")
        elif sys.platform == 'darwin':  # macOS
            # AMD GPUs are rare on Mac, but check system_profiler
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                amd_detected = True
                print("✅ AMD GPU detected in system (via system_profiler)")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    if amd_detected:
        print("  💡 AMD GPU Support Status:")
        
        # Check ROCm installation status
        if rocm_path:
            print(f"    ROCM_PATH: {rocm_path}")
        if hip_visible:
            print(f"    HIP_VISIBLE_DEVICES: {hip_visible}")
        else:
            print("    HIP_VISIBLE_DEVICES: Not set (will use all devices)")
        
        # Check if PyTorch ROCm is installed
        try:
            import torch_rocm
            print("    ✅ PyTorch ROCm support available")
            print("    💡 Recommendation: AMD GPU training possible")
            print("    💡 Suggested command: --device auto (with ROCm PyTorch)")
        except ImportError:
            print("    ⚠️  PyTorch ROCm not installed")
            print("    💡 To install: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1")
            print("    💡 Or use CPU fallback: --cpu-mode")
        
        # Check for specific AMD GPU models
        print("  💡 AMD GPU Support Notes:")
        print("    - AMD Instinct MI200/MI300 series: Full ROCm support")
        print("    - AMD Radeon RX 6000/7000 series: Limited ROCm support")
        print("    - AMD MX300x series: Not officially supported by ROCm")
        print("    - For MX300x: CPU fallback recommended")
        print("    - Older AMD GPUs: CPU fallback recommended")
        
        return True
    else:
        print("❌ No AMD GPU detected")
        print("  💡 If you have an AMD GPU:")
        print("    - Install ROCm: https://rocm.docs.amd.com/")
        print("    - Install PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1")
        print("    - Set HIP_VISIBLE_DEVICES environment variable")
        
        return False

def check_amd_quick():
    """Quick AMD GPU check without printing (for recommendations)"""
    try:
        # Check environment variables
        if os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('ROCM_PATH'):
            return True
        
        # Quick rocm-smi check
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            return True
    except:
        pass
    
    return False

def recommend_settings():
    """Provide overall recommendations"""
    print("\n🎯 Training Recommendations:")
    print("-" * 30)
    
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    has_amd = check_amd_quick()  # Use quick check to avoid duplicate output
    
    if has_cuda:
        print("🏆 BEST: Use NVIDIA GPU for fastest training")
        print("   Command: python text_to_sql_train.py --device cuda")
        
        # Check if high-end GPU
        if has_cuda:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                print("   🚀 High-end GPU detected - can handle large datasets")
                print("   Command: python text_to_sql_train.py --device cuda --max-samples 10000")
                
    elif has_amd:
        print("🥈 GOOD: AMD GPU detected - ROCm training possible")
        print("   Command: python text_to_sql_train.py --device auto")
        print("   💡 Ensure ROCm PyTorch is installed for optimal performance")
        
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
            "name": "AMD GPU (ROCm)",
            "command": "python text_to_sql_train.py --device auto --max-samples 5000",
            "description": "AMD GPU with ROCm acceleration"
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
    has_amd = check_amd()
    
    # Show recommendations
    recommend_settings()
    
    # Show example commands
    show_example_commands()
    
    print("🎉 Device check complete!")
    print("💡 Choose the command that matches your hardware capabilities.")

if __name__ == "__main__":
    main()
