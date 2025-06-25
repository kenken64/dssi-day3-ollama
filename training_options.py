#!/usr/bin/env python3
"""
Text-to-SQL Training Options Summary
Shows all available device and configuration options
"""

def show_device_options():
    print("🖥️ Device Options:")
    print("-" * 20)
    print("--device auto     : Auto-detect best device (default)")
    print("--device cuda     : Force NVIDIA GPU")
    print("--device mps      : Force Apple Silicon MPS")
    print("--device amd      : Force AMD GPU (ROCm required)")
    print("--device cpu      : Force CPU")
    print("--cpu-mode        : Enable CPU optimizations")
    print("--amd-mode        : Enable AMD GPU optimizations")
    print("--fast-mac        : Aggressive Mac optimizations")
    print("--no-quantization : Disable 4-bit quantization")

def show_model_options():
    print("\n🤖 Model Options:")
    print("-" * 18)
    print("Default: codellama/CodeLlama-7b-Instruct-hf (7GB)")
    print("--base-model Salesforce/codet5p-220m    : Tiny model (220MB)")
    print("--base-model Salesforce/codet5p-770m    : Small model (770MB)")
    print("--base-model defog/sqlcoder-7b          : SQL specialist (7GB)")
    print("--base-model microsoft/CodeBERT-base    : General code model")

def show_training_options():
    print("\n⚙️ Training Options:")
    print("-" * 20)
    print("--max-samples 1000    : Limit training samples")
    print("--save-full-model     : Save complete model (not just adapter)")
    print("--model-name custom   : Custom Ollama model name")

def show_mode_options():
    print("\n🎯 Execution Modes:")
    print("-" * 19)
    print("--mode full    : Complete pipeline (train + deploy + test)")
    print("--mode train   : Training only")
    print("--mode deploy  : Deploy existing model to Ollama")
    print("--mode test    : Test deployed model")

def show_examples():
    print("\n📝 Example Commands:")
    print("-" * 20)
    
    examples = [
        ("Quick Start (Auto-detect)", "python text_to_sql_train.py"),
        ("NVIDIA GPU", "python text_to_sql_train.py --device cuda"),
        ("Apple Silicon", "python text_to_sql_train.py --device mps"),
        ("CPU Only", "python text_to_sql_train.py --cpu-mode"),
        ("Fast Mac Test", "python text_to_sql_train.py --fast-mac"),
        ("Tiny Model", "python text_to_sql_train.py --base-model Salesforce/codet5p-220m"),
        ("Large Dataset (GPU)", "python text_to_sql_train.py --device cuda --max-samples 10000"),
        ("Small Dataset (CPU)", "python text_to_sql_train.py --cpu-mode --max-samples 250"),
        ("Training Only", "python text_to_sql_train.py --mode train --device cuda"),
        ("Custom Model Name", "python text_to_sql_train.py --model-name my-sql-model"),
    ]
    
    for name, command in examples:
        print(f"\n{name}:")
        print(f"  {command}")

def show_device_recommendations():
    print("\n💡 Device Recommendations:")
    print("-" * 30)
    print("🏆 RTX 4090/4080: --device cuda --max-samples 20000")
    print("🥇 RTX 3080/3090: --device cuda --max-samples 10000")
    print("🥈 MI300X/MI250: --device amd --max-samples 8000")
    print("🥈 Apple M2 Max:  --device mps --max-samples 5000")
    print("🥈 Apple M1/M2:   --device mps --max-samples 2000")
    print("🥉 AMD RX 7900:  --device amd --max-samples 3000")
    print("🥉 AMD MX300x:   --cpu-mode --max-samples 500")
    print("🥉 Intel 16-core: --cpu-mode --max-samples 500")
    print("🥉 Intel 8-core:  --cpu-mode --max-samples 250")

def show_troubleshooting():
    print("\n🔧 Troubleshooting:")
    print("-" * 19)
    print("Out of memory      → Reduce --max-samples or use smaller model")
    print("CUDA issues        → Try --no-quantization")
    print("MPS issues         → Use --device cpu as fallback")
    print("AMD/ROCm issues    → Install ROCm PyTorch or use --cpu-mode")
    print("MX300x not working → Use --cpu-mode (MX300x not ROCm supported)")
    print("Slow training      → Check device with: python check_device.py")
    print("Model not found    → Ensure base model name is correct")
    print("Token loops        → Run: python fix_ollama_tokens.py")

def main():
    print("🤖 Text-to-SQL Training Options Summary")
    print("=" * 50)
    
    show_device_options()
    show_model_options()
    show_training_options()
    show_mode_options()
    show_examples()
    show_device_recommendations()
    show_troubleshooting()
    
    print("\n🎉 For detailed device info: python check_device.py")
    print("🌐 For web interface: python start_webapp.py")
    print("📚 For documentation: see README.md")

if __name__ == "__main__":
    main()
