# Text-to-SQL Ollama Model

A comprehensive pipeline for training and deploying text-to-SQL models using the Gretel AI synthetic dataset and Ollama for local deployment.

## 🚀 Features

- **Complete Training Pipeline**: Fine-tune models on the Gretel AI synthetic text-to-SQL dataset
- **Multiple Base Models**: Support for CodeLlama, SQLCoder, and other popular models
- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- **Ollama Integration**: Deploy models locally with Ollama for fast inference
- **Comprehensive Testing**: Built-in test suite with performance benchmarks
- **Domain Coverage**: Supports 25+ domains including healthcare, finance, e-commerce
- **SQL Complexity**: Handles basic queries to complex joins, subqueries, and window functions

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM (32GB recommended for larger models)
- 50GB+ free disk space

### Software Dependencies
- [Ollama](https://ollama.ai/) - For model deployment
- PyTorch 2.0+
- Transformers 4.36+
- CUDA Toolkit (for GPU training)

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

### Option 1: Full Pipeline (Recommended)
```bash
# Train and deploy the model in one command
python text_to_sql_train.py --mode full --max-samples 1000

# This will:
# 1. Download and process the Gretel AI dataset
# 2. Fine-tune the model using LoRA
# 3. Deploy to Ollama
# 4. Run basic tests
```

### Option 2: Step-by-Step

#### Step 1: Train the Model
```bash
python text_to_sql_train.py --mode train --max-samples 5000
```

#### Step 2: Deploy to Ollama
```bash
python deploy.py --config config.yaml
```

#### Step 3: Test the Deployment
```bash
python test.py --config config.yaml
```

### Option 3: Use Pre-configured Models
```bash
# Use SQLCoder-7B (recommended for SQL tasks)
python text_to_sql_train.py --mode full --base-model "defog/sqlcoder-7b"

# Use CodeLlama (good general performance)
python text_to_sql_train.py --mode full --base-model "codellama/CodeLlama-7b-Instruct-hf"
```

## 📊 Usage Examples

### Basic Query Generation
```bash
ollama run text-to-sql "Database Schema:
CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(30), salary DECIMAL(10,2));

Request: Find all employees in the Engineering department with salary above 75000"
```

### Interactive Demo
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

### Basic Configuration (config.yaml)
```yaml
model:
  base_model_name: "defog/sqlcoder-7b"
  
training:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3
  max_length: 1024

ollama:
  model_name: "text-to-sql"
  temperature: 0.1
```

### Advanced Configuration
```yaml
# For larger models
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  
quantization:
  use_4bit: true
  
# For better performance
lora:
  r: 16
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
| Model | Syntax Accuracy | Execution Accuracy | Avg Response Time |
|-------|----------------|--------------------|-------------------|
| SQLCoder-7B | 95%+ | 85%+ | 2-4s |
| CodeLlama-7B | 90%+ | 80%+ | 3-5s |
| CodeLlama-13B | 97%+ | 90%+ | 5-8s |

### Optimization Tips
1. **Use SQLCoder models** for best SQL performance
2. **Increase LoRA rank** (r=16-32) for better quality
3. **Use quantization** to reduce memory usage
4. **Batch inference** for multiple queries

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

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python text_to_sql_train.py --batch-size 2 --gradient-accumulation-steps 8

# Use 4-bit quantization
# Set use_4bit: true in config.yaml
```

#### 2. Model Not Found in Ollama
```bash
# Check available models
ollama list

# Recreate model
python deploy.py --force
```

#### 3. Poor SQL Quality
```bash
# Increase training data
python text_to_sql_train.py --max-samples 10000

# Use specialized model
# Set base_model_name: "defog/sqlcoder-7b" in config.yaml

# Increase LoRA parameters
# Set r: 16, alpha: 64 in config.yaml
```

#### 4. Slow Training
```bash
# Enable gradient checkpointing
# Set gradient_checkpointing: true in config.yaml

# Use mixed precision
# Set fp16: true in training arguments

# Reduce sequence length
# Set max_length: 512 in config.yaml
```

### Performance Optimization

#### Memory Optimization
```yaml
quantization:
  use_4bit: true
  bnb_4bit_compute_dtype: "float16"

training:
  gradient_checkpointing: true
  dataloader_drop_last: true
```

#### Speed Optimization
```yaml
hardware:
  use_flash_attention: true  # If available
  torch_dtype: "float16"

training:
  dataloader_num_workers: 4
```

## 📚 Dataset Information

### Gretel AI Synthetic Text-to-SQL Dataset
- **Size**: 100K+ examples
- **Domains**: 25+ including healthcare, finance, cybersecurity
- **Complexity Levels**: Basic SQL, aggregations, joins, subqueries, window functions
- **Quality**: High-quality synthetic data with explanations

### Supported SQL Features
- SELECT, INSERT, UPDATE, DELETE statements
- JOINs (INNER, LEFT, RIGHT, FULL)
- Aggregations (COUNT, SUM, AVG, MIN, MAX)
- GROUP BY, HAVING, ORDER BY
- Subqueries and CTEs
- Window functions
- Date/time operations
- String functions

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Testing Guidelines
- Add unit tests for new functions
- Include integration tests for major features
- Test with different model sizes
- Validate SQL output quality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Gretel AI** for the high-quality synthetic text-to-SQL dataset
- **Ollama** for the excellent local model deployment platform
- **Hugging Face** for the transformers library and model hub
- **Microsoft** for the LoRA implementation in PEFT

## 📞 Support

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/yourusername/text-to-sql-ollama/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/text-to-sql-ollama/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/text-to-sql-ollama/wiki)

### Commercial Support
For commercial support, training, or custom implementations, please contact [your.email@example.com](mailto:your.email@example.com).

## 🗺️ Roadmap

### Upcoming Features
- [ ] Support for more SQL dialects (PostgreSQL, MySQL, SQL Server)
- [ ] Web interface for model interaction
- [ ] API server for production deployment
- [ ] Integration with popular databases
- [ ] Query optimization suggestions
- [ ] Multi-language support
- [ ] Federated learning capabilities

### Version History
- **v1.0.0**: Initial release with basic training and deployment
- **v1.1.0**: Added comprehensive testing suite
- **v1.2.0**: Docker support and performance optimizations
- **v2.0.0**: Multi-model support and advanced features (planned)

---

**Made with ❤️ for the open source community**