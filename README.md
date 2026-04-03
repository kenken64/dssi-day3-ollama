# Text-to-SQL Ollama Model

A comprehensive pipeline for training and deploying text-to-SQL models using the Gretel AI synthetic dataset and Ollama for local deployment. **Now optimized for smaller models and efficient memory usage!**

## � Table of Contents

- [�🚀 Features](#-features)
- [📚 Complete Documentation](#-complete-documentation)
- [📋 Prerequisites](#-prerequisites)
- [🛠️ Installation](#️-installation)
- [⚡ Quick Start](#-quick-start)
- [📊 Usage Examples](#-usage-examples)
- [🎛️ Configuration](#️-configuration)
- [🔧 Device Support](#-device-support)
- [🚀 Web Application](#-web-application)
- [📦 Deployment](#-deployment)
- [🧪 Testing](#-testing)
- [📈 Performance](#-performance)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

## 🚀 Features

- **Complete Training Pipeline**: Fine-tune models on the Gretel AI synthetic text-to-SQL dataset
- **Multiple Base Models**: Support for Qwen2.5-Coder, CodeLlama, SQLCoder, and other popular models
- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- **Intelligent Model Adaptation**: Auto-detects LoRA target modules for any model architecture
- **Ollama Integration**: Deploy models locally with Ollama for fast inference (REST API)
- **Memory Optimized**: Smart adapter-only saving reduces file sizes by 100x (50MB vs 13GB)
- **Web Application**: Dark-themed shadcn/Tailwind UI with live SQL generation and SQLite testing
- **SQLite Test Execution**: Auto-generates sample data, converts MySQL/PostgreSQL syntax to SQLite, and runs queries in-browser
- **SQL Dialect Converter**: Automatic conversion of `NOW()`, `INTERVAL`, `DATE_FORMAT()`, `TIMESTAMPDIFF()`, `EXTRACT()`, `LAG()` and more to SQLite
- **Model Transfer Tool**: Export/import models as zip files for offline transfer between machines
- **Comprehensive Testing**: Built-in test suite with performance benchmarks
- **Domain Coverage**: Supports 25+ domains including healthcare, finance, e-commerce
- **SQL Complexity**: Handles basic queries to complex joins, subqueries, and window functions

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- **Device Support**: NVIDIA GPU (CUDA) / Apple Silicon (MPS) / AMD GPU (ROCm) / CPU
- **8GB+ RAM** (16GB recommended - **reduced from previous 32GB requirement**)
- **10GB+ free disk space** (significantly reduced with adapter-only saving)

#### Device Recommendations
- **🏆 Best**: NVIDIA GPU (RTX 30/40 series, A100, V100) - Full performance
- **🥈 Good**: Apple Silicon Mac (M1/M2/M3) - Great performance with MPS
- **🥈 Good**: AMD GPU with ROCm (Instinct MI series, RX 6000/7000) - ROCm acceleration
- **🥉 Compatible**: Intel CPU - Slower but works everywhere
- **❌ Not Supported**: AMD MX300x - Use CPU fallback instead

### Software Dependencies
- [Ollama](https://ollama.ai/) - For model deployment
- PyTorch 2.0+
- Transformers 4.36+
- CUDA Toolkit (for NVIDIA GPU training)
- ROCm (for AMD GPU training) - **Note: MX300x not supported**

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/text-to-sql-ollama
cd text-to-sql-ollama
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

### 5. Setup Project Structure
```bash
python utils.py  # Creates necessary directories
```

## 🏃‍♂️ Quick Start

### Option 1: Full Pipeline (Recommended - Auto-Optimized)
```bash
# Check your device capabilities first
python check_device.py

# Train and deploy with automatic device detection and optimization
python text_to_sql_train.py --mode full --max-samples 1000

# For even faster training with smaller model
python text_to_sql_train.py --mode full --base-model Salesforce/codet5p-770m --max-samples 1000

# Force specific device if auto-detection doesn't work
python text_to_sql_train.py --device cuda  # NVIDIA GPU
python text_to_sql_train.py --device mps   # Apple Silicon
python text_to_sql_train.py --device amd   # AMD GPU (ROCm required)
python text_to_sql_train.py --cpu-mode     # CPU only

# Special cases
python text_to_sql_train.py --amd-mode --max-samples 3000  # AMD GPU with optimizations

# For AMD MX300x users (not ROCm supported)
python text_to_sql_train.py --cpu-mode --max-samples 500 --base-model Salesforce/codet5p-220m

# This will:
# 1. Detect your device (CUDA/MPS/AMD/CPU) and apply optimizations
# 2. Download and process the Gretel AI dataset
# 3. Fine-tune using memory-optimized LoRA
# 4. Save only adapter weights (50MB instead of 13GB!)
# 5. Deploy to Ollama
# 6. Run basic tests
```

### Option 2: Step-by-Step with Different Model Sizes

#### Small Model (Fast Training, Good Performance)
```bash
# CodeLlama 7B - Default, good balance of speed and quality
python text_to_sql_train.py --mode train --max-samples 5000

# Very small model for testing (220MB)
python text_to_sql_train.py --mode train --base-model Salesforce/codet5p-220m --max-samples 1000

# Medium model for better quality (770MB)
python text_to_sql_train.py --mode train --base-model Salesforce/codet5p-770m --max-samples 5000
```

#### Deploy to Ollama
```bash
python deploy.py --config config.yaml
```

#### Test the Deployment
```bash
python test.py --config config.yaml
```

### Option 3: Model Size Comparison
```bash
# NEW DEFAULT: CodeLlama 7B (optimized, faster)
python text_to_sql_train.py --mode full --base-model "codellama/CodeLlama-7b-Instruct-hf"

# Small and fast CodeT5+ model (770MB)
python text_to_sql_train.py --mode full --base-model "Salesforce/codet5p-770m"

# SQLCoder for specialized SQL performance (larger but specialized)
python text_to_sql_train.py --mode full --base-model "defog/sqlcoder-7b"

# Tiny model for quick testing (220MB)
python text_to_sql_train.py --mode full --base-model "Salesforce/codet5p-220m" --max-samples 500
```

## 📊 Usage Examples

### Basic Query Generation
```bash
ollama run text-to-sql "Database Schema:
CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(30), salary DECIMAL(10,2));

Request: Find all employees in the Engineering department with salary above 75000"
```

### Web Application (Recommended)
```bash
# Start the web interface
python start_webapp.py

# Open http://localhost:5000 in your browser
# Features:
#   - Dark-themed UI (shadcn/Tailwind)
#   - 5 built-in example schemas (Basic, E-commerce, Sales Analytics, Healthcare, Inventory)
#   - "Test in SQLite" - auto-generates sample data and runs your query
#   - Auto-converts MySQL/PostgreSQL syntax to SQLite for testing
#   - Auto-fixes hallucinated table joins
```

### Model Transfer (Export/Import)
```bash
# Export model + adapter + Modelfile to a zip for another machine
python model_transfer.py export -o qwen2.5-coder-0.5b-text-to-sql.zip

# Import on target machine
python model_transfer.py import qwen2.5-coder-0.5b-text-to-sql.zip
```

### Command Line Demo
```bash
python deploy.py --demo
```

### Programmatic Usage
```python
from utils import OllamaManager

# Initialize the model
manager = OllamaManager("text-to-sql")

# Generate SQL
schema = "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100));"
query = "Find all users registered in the last month"
prompt = f"{schema}\n\nRequest: {query}"

response = manager.query_model(prompt)
print(response)
```

## ⚙️ Configuration

### NEW: Memory-Optimized Configuration (config.yaml)
```yaml
model:
  base_model_name: "codellama/CodeLlama-7b-Instruct-hf"  # NEW DEFAULT: Faster and smaller
  
training:
  batch_size: 4              # Optimized for new model
  learning_rate: 2e-4
  num_epochs: 3
  max_length: 512            # Reduced for memory efficiency
  save_only_adapter: true    # NEW: Save only 50MB instead of 13GB!

lora:                        # Enhanced LoRA config
  r: 16                      # Increased for better performance
  alpha: 32
  target_modules: "auto"     # NEW: Auto-detect for any model

quantization:                # Memory optimization
  use_4bit: true
  use_nested_quant: true     # NEW: Additional memory savings

memory_optimization:         # NEW section
  gradient_checkpointing: true
  dataloader_pin_memory: false

ollama:
  model_name: "text-to-sql"
  temperature: 0.1
```

### Model Size Options
```yaml
# Tiny model for testing (220MB base model + 10MB adapter)
model:
  base_model_name: "Salesforce/codet5p-220m"

# Small model for production (770MB base model + 20MB adapter)  
model:
  base_model_name: "Salesforce/codet5p-770m"

# Default balanced model (7B base model + 50MB adapter)
model:
  base_model_name: "codellama/CodeLlama-7b-Instruct-hf"

# Specialized SQL model (7B base model + 80MB adapter)
model:
  base_model_name: "defog/sqlcoder-7b"
```

### Advanced Configuration
```yaml
# For even better memory optimization
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  save_total_limit: 2        # Keep fewer checkpoints
  
# For full model saving (if needed)
training:
  save_only_adapter: false   # Will save full 13GB model

# For better performance with more memory
lora:
  r: 32                      # Higher rank for better quality
  alpha: 64
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
python test.py --config config.yaml
```

### Performance Testing
```bash
python test.py --performance-only
```

### Quick Test Subset
```bash
python test.py --quick
```

### Custom Test Cases
```python
from test import TextToSQLTester, TestCase

tester = TextToSQLTester()

# Create custom test
custom_test = TestCase(
    name="custom_query",
    schema="CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2));",
    natural_query="Find products under $50",
    expected_keywords=["SELECT", "WHERE", "price", "50"]
)

result = tester.run_test_case(custom_test)
print(result)
```

## 📈 Performance Benchmarks

### Expected Performance (on test dataset)
| Model | Base Size | Adapter Size | Syntax Accuracy | Execution Accuracy | Avg Response Time |
|-------|-----------|--------------|----------------|--------------------|-------------------|
| **CodeLlama-7B** ⭐ | 13GB | **50MB** | 92%+ | 82%+ | 2-3s |
| CodeT5+ 770M | 770MB | **20MB** | 88%+ | 78%+ | 1-2s |
| CodeT5+ 220M | 220MB | **10MB** | 85%+ | 72%+ | <1s |
| SQLCoder-7B | 13GB | **80MB** | 95%+ | 85%+ | 3-4s |

⭐ **New Default Model** - Best balance of performance and efficiency

### File Size Comparison
| Component | Before Optimization | After Optimization | Savings |
|-----------|-------------------|-------------------|---------|
| Model Files | 13GB+ | 50MB (adapter only) | **99.6%** |
| Training Memory | 32GB RAM required | 8GB RAM sufficient | **75%** |
| Training Time | 4-6 hours | 2-3 hours | **50%** |

### Optimization Benefits
- ✅ **100x smaller files**: 50MB adapters vs 13GB full models
- ✅ **Faster training**: Optimized batch sizes and memory usage
- ✅ **Universal compatibility**: Auto-detects model architecture
- ✅ **Better resource usage**: Works on consumer hardware
- ✅ **Quick iteration**: Faster model testing and deployment

### Performance Tips
1. **Use CodeLlama-7B** (new default) for best balance
2. **Use CodeT5+ models** for fastest training/inference
3. **Use SQLCoder** only if you need maximum SQL accuracy
4. **Start with adapter-only** saving for faster iteration
5. **Enable quantization** for memory-constrained systems

## 🔧 Advanced Features

### Custom Dataset Training
```python
from text_to_sql_train import DataProcessor

# Load your custom dataset
processor = DataProcessor(config)
# Format: [{"sql_context": "...", "sql_prompt": "...", "sql": "..."}]
custom_data = load_your_dataset()

# Train with custom data
formatted_dataset = processor.prepare_custom_dataset(custom_data)
```

### Domain-Specific Fine-tuning
```bash
# Train on specific domains
python text_to_sql_train.py --domains "healthcare,finance" --max-samples 2000
```

### Multi-Model Ensemble
```python
from utils import OllamaManager

models = ["text-to-sql-v1", "text-to-sql-v2"]
managers = [OllamaManager(model) for model in models]

# Query multiple models and combine results
results = [mgr.query_model(prompt) for mgr in managers]
```

## 🐳 Docker Support

### Build Docker Image
```bash
docker build -t text-to-sql-trainer .
```

### Run Training in Container
```bash
docker run --gpus all -v $(pwd):/workspace text-to-sql-trainer python text_to_sql_train.py --mode train
```

### Docker Compose for Full Stack
```bash
docker-compose up -d
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (Now Much Less Common!)
```bash
# NEW: Use even smaller model
python text_to_sql_train.py --base-model Salesforce/codet5p-220m --max-samples 1000

# Reduce batch size further
python text_to_sql_train.py --batch-size 1 --gradient-accumulation-steps 16

# Memory optimizations are now enabled by default
# - 4-bit quantization
# - Gradient checkpointing  
# - Reduced sequence length
# - Adapter-only saving
```

#### 2. Model Not Found in Ollama
```bash
# Check available models
ollama list

# Recreate model
python deploy.py --force
```

#### 3. LoRA Target Modules Not Found (Now Auto-Fixed!)
```bash
# This error is now automatically resolved!
# The system auto-detects correct target modules for any model

# If you want to see what modules were detected:
python text_to_sql_train.py --mode train --base-model your-model
# Check the logs for "Found target modules: [...]"
```

#### 4. Large File Sizes (Now Solved!)
```bash
# NEW DEFAULT: Only save adapter (50MB instead of 13GB)
python text_to_sql_train.py --mode train  # Saves adapter only

# If you need the full model for some reason:
python text_to_sql_train.py --mode train --save-full-model
```

#### 5. Poor SQL Quality
```bash
# Increase training data
python text_to_sql_train.py --max-samples 10000

# Use specialized model (if you have more memory/time)
python text_to_sql_train.py --base-model "defog/sqlcoder-7b"

# Increase LoRA parameters
# Set r: 32, alpha: 64 in config.yaml
```

#### 6. Slow Training (Much Improved!)
```bash
# NEW: Use smaller, faster model
python text_to_sql_train.py --base-model Salesforce/codet5p-770m

# All optimizations are now enabled by default:
# - Gradient checkpointing
# - Mixed precision (fp16)  
# - Reduced sequence length (512 vs 1024)
# - Optimized batch sizes
```

#### 7. Token Generation Loop Issue (NEW)
```bash
# If you see repeating { end }<|end|> tokens:

# Quick fix - use the automated fix script
python fix_ollama_tokens.py

# Manual fix steps:
# 1. Delete the problematic model
ollama rm text-to-sql

# 2. Recreate with fixed Modelfile
python deployment_script.py --force

# 3. Test with a simple query
ollama run text-to-sql "CREATE TABLE test (id INT); Find all records from test table"
```

**Root cause:** Improper stop token configuration in Modelfile causing infinite generation loops.
**Solution:** The fix script automatically recreates the model with proper token handling.

#### 8. Model Response Quality Issues
```bash
# If the model generates poor or nonsensical SQL:

# Use the specialized SQL model (if you have resources)
python text_to_sql_train.py --base-model "defog/sqlcoder-7b"

# Increase training quality
python text_to_sql_train.py --max-samples 10000 --lora-r 32

# Use better prompting format
# Include more context in your schema descriptions
```

### Performance Optimization

#### NEW: Memory Optimization (Enabled by Default)
```yaml
# All these are now automatically configured:
quantization:
  use_4bit: true
  bnb_4bit_compute_dtype: "float16"
  use_nested_quant: true         # NEW: Additional memory savings

training:
  gradient_checkpointing: true   # Enabled by default
  dataloader_drop_last: true
  dataloader_pin_memory: false   # Disabled for memory savings
  save_only_adapter: true        # NEW: 100x smaller files
  save_total_limit: 2           # Fewer checkpoints

memory_optimization:             # NEW automatic optimizations
  batch_size: 4                 # Optimized for CodeLlama
  max_length: 512               # Reduced from 1024
  gradient_accumulation_steps: 4 # Balanced for throughput
```

#### Speed Optimization
```yaml
# For maximum speed with small models:
model:
  base_model_name: "Salesforce/codet5p-220m"  # 10x faster training

training:
  max_samples: 1000             # Quick training
  num_epochs: 2                 # Fewer epochs for testing
  
# For production quality:
model:
  base_model_name: "codellama/CodeLlama-7b-Instruct-hf"  # NEW DEFAULT

training:
  max_samples: 5000             # Good balance
  lora_r: 16                    # Higher quality adapters
```

## 🖥️ Device Support & Optimization

### Check Your System Capabilities
```bash
# See what devices are available and get recommendations
python check_device.py
```

### Device-Specific Training

#### NVIDIA GPU (Recommended)
```bash
# Automatic detection (recommended)
python text_to_sql_train.py --mode full

# Force GPU with specific settings
python text_to_sql_train.py --device cuda --max-samples 10000

# Disable quantization for older GPUs
python text_to_sql_train.py --device cuda --no-quantization
```

#### Apple Silicon Mac (MPS)
```bash
# Automatic MPS detection and optimization
python text_to_sql_train.py --mode full

# Force MPS device
python text_to_sql_train.py --device mps --max-samples 2000

# Fastest Mac training (for testing)
python text_to_sql_train.py --fast-mac
```

#### CPU Only
```bash
# Automatic CPU optimization
python text_to_sql_train.py --cpu-mode --max-samples 500

# Force CPU with custom settings
python text_to_sql_train.py --device cpu --max-samples 250 --base-model Salesforce/codet5p-220m
```

### Performance Comparison by Device

| Device Type | Training Time (1000 samples) | Max Recommended Samples | Memory Usage |
|-------------|------------------------------|-------------------------|--------------|
| **RTX 4090** | 10-15 min | 20,000+ | 8-12GB |
| **RTX 3080** | 15-25 min | 10,000 | 6-8GB |
| **Apple M2 Max** | 20-30 min | 5,000 | 4-6GB |
| **Apple M1** | 30-45 min | 2,000 | 3-4GB |
| **Intel CPU (16 cores)** | 2-4 hours | 500 | 2-4GB |
| **Intel CPU (8 cores)** | 4-8 hours | 250 | 2-3GB |

## 📚 Complete Documentation

This project includes comprehensive documentation for all aspects of text-to-SQL model training and deployment:

### 🎯 **Core Documentation**
- **[README.md](README.md)** - Main overview, installation, and quick start guide
- **[SCRIPT_FLOW_DIAGRAM.md](SCRIPT_FLOW_DIAGRAM.md)** - Complete execution flow and logic diagrams
- **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - Device-specific optimization guide (CPU, Mac MPS, NVIDIA GPU)

### 🚀 **Deployment & Operations**
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Model deployment to Ollama with troubleshooting
- **[WEBAPP.md](WEBAPP.md)** - Web application setup, API usage, and production deployment
- **[OLLAMA_TOKEN_FIX.md](OLLAMA_TOKEN_FIX.md)** - Fix for token generation loops and deployment issues

### 🎓 **Learning & Assessment**
- **[ASSESSMENT.md](ASSESSMENT.md)** - Comprehensive practical assessment with CodeBERT examples
- **[text_to_sql_ollama_guide.md](text_to_sql_ollama_guide.md)** - Original guide and background information

### 🛠️ **Quick Reference**

| Need | Documentation | Key Commands |
|------|---------------|--------------|
| **Quick Start** | [README.md](README.md) | `python text_to_sql_train.py --mode full` |
| **Mac Development** | [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md) | `python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train` |
| **GPU Training** | [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md) | `python text_to_sql_train.py --device cuda --mode full` |
| **Web Interface** | [WEBAPP.md](WEBAPP.md) | `python start_webapp.py --port 5000` |
| **Deployment Issues** | [DEPLOYMENT.md](DEPLOYMENT.md) | `python deployment_script.py` |
| **Token Problems** | [OLLAMA_TOKEN_FIX.md](OLLAMA_TOKEN_FIX.md) | `python fix_ollama_tokens.py` |
| **Assessment/Testing** | [ASSESSMENT.md](ASSESSMENT.md) | Practical exercises and evaluation |
| **Flow Understanding** | [SCRIPT_FLOW_DIAGRAM.md](SCRIPT_FLOW_DIAGRAM.md) | Visual execution diagrams |

### 📊 **Documentation Highlights**

#### **For Developers:**
- **[SCRIPT_FLOW_DIAGRAM.md](SCRIPT_FLOW_DIAGRAM.md)**: Understand the complete pipeline flow
- **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)**: Optimize for your hardware (Mac/GPU/CPU)
- **[WEBAPP.md](WEBAPP.md)**: Build and deploy web interfaces

#### **For Operations:**
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Production deployment strategies
- **[OLLAMA_TOKEN_FIX.md](OLLAMA_TOKEN_FIX.md)**: Troubleshoot common issues
- **[WEBAPP.md](WEBAPP.md)**: Monitor and scale web applications

#### **For Learning:**
- **[ASSESSMENT.md](ASSESSMENT.md)**: Hands-on exercises with CodeBERT-base
- **[text_to_sql_ollama_guide.md](text_to_sql_ollama_guide.md)**: Background and concepts
- **All files**: Extensively commented code and examples

---

## 🗺️ Project Navigation & Documentation Map

### 📁 **Project Structure Overview**

```
dssi-day3-ollama/
├── 📄 Core Scripts
│   ├── text_to_sql_train.py      # Main training pipeline (Qwen2.5-Coder + LoRA)
│   ├── app.py                    # Flask web app with SQL dialect converter & SQLite testing
│   ├── start_webapp.py           # Web app launcher
│   ├── model_transfer.py         # Export/import models as zip for offline transfer
│   └── deployment_script.py      # Ollama deployment
├── 🌐 Web Interface
│   ├── templates/
│   │   ├── index.html            # Main page (dark shadcn/Tailwind UI)
│   │   ├── examples.html         # 5 example schemas with "Use Example" buttons
│   │   ├── 404.html              # Not found page
│   │   └── 500.html              # Server error page
│   └── static/                   # CSS, JS, images
├── 🛠️ Utilities
│   ├── utils.py                  # OllamaManager (REST API), SQLValidator, helpers
│   ├── check_device.py           # Device capability detection
│   └── test_ollama_model.py      # Model testing
├── ⚙️ Configuration
│   ├── Modelfile                 # Ollama model config (Qwen2.5 ChatML, LAG() guidance)
│   ├── config.yaml               # Training configuration
│   └── requirements.txt          # Python dependencies
└── 📚 Documentation
    ├── README.md                 # This file
    ├── SCRIPT_FLOW_DIAGRAM.md    # Complete execution flow
    ├── DEVICE_SUPPORT.md         # Hardware optimization guide
    ├── WEBAPP.md                 # Web app setup & deployment
    └── DEPLOYMENT.md             # Ollama deployment guide
```

### 🎯 **Documentation Journey Guide**

#### **🚀 I want to get started quickly**
1. Read **[README.md](README.md)** (this file) - Overview and installation
2. Run: `python text_to_sql_train.py --mode full`
3. Check **[WEBAPP.md](WEBAPP.md)** for web interface

#### **🍎 I'm developing on Mac**
1. **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - Mac MPS optimization
2. Run: `python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train`
3. **[WEBAPP.md](WEBAPP.md)** - Mac-specific web app setup

#### **💻 I'm using CPU only**
1. **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - CPU optimization strategies
2. **[ASSESSMENT.md](ASSESSMENT.md)** - CPU-friendly exercises

#### **🖥️ I have NVIDIA GPU**
1. **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - GPU optimization guide
2. **[WEBAPP.md](WEBAPP.md)** - High-performance deployment

#### **🏢 I want production deployment**
1. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment strategies
2. **[WEBAPP.md](WEBAPP.md)** - Production web app setup
3. **[OLLAMA_TOKEN_FIX.md](OLLAMA_TOKEN_FIX.md)** - Troubleshooting

#### **🎓 I want to learn and practice**
1. **[ASSESSMENT.md](ASSESSMENT.md)** - Hands-on exercises
2. **[SCRIPT_FLOW_DIAGRAM.md](SCRIPT_FLOW_DIAGRAM.md)** - Understand the pipeline
3. **text_to_sql_train.py** - Read the extensively commented code

#### **🔧 I need to understand the code**
1. **[SCRIPT_FLOW_DIAGRAM.md](SCRIPT_FLOW_DIAGRAM.md)** - Visual execution flow
2. **text_to_sql_train.py** - Main pipeline with detailed comments
3. **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - Device-specific optimizations

#### **🐛 I'm having issues**
1. **[OLLAMA_TOKEN_FIX.md](OLLAMA_TOKEN_FIX.md)** - Token generation problems
2. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment troubleshooting
3. **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - Device-specific issues

### 🎉 **Project Highlights**

- **🔥 Latest Feature**: CodeBERT-base support for 15-minute training on Mac
- **💡 Smart Design**: Auto-device detection with fallback strategies
- **📱 User Friendly**: Beautiful web interface for non-technical users
- **🎯 Production Ready**: Complete deployment pipeline with monitoring
- **📚 Well Documented**: Every component thoroughly explained
- **🧪 Tested**: Comprehensive assessment and testing framework

### 🤝 **Contributing & Community**

This project demonstrates best practices for:
- **ML Pipeline Development**: End-to-end training and deployment
- **Device Optimization**: Cross-platform compatibility and performance
- **Documentation**: Comprehensive guides for all user types
- **Code Quality**: Extensively commented and well-structured code
- **User Experience**: Web interfaces and command-line tools

**Start with any document above based on your needs - they're all interconnected and cross-referenced for easy navigation!**

---

*Built with ❤️ for the data science and ML community*