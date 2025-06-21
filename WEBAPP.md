# Text-to-SQL Web Application

A modern, user-friendly web interface for the Text-to-SQL model powered by Flask and Ollama, supporting multiple base models including CodeBERT-base for fast prototyping.

## 🌟 Features

- **Beautiful Interface**: Modern, responsive web design
- **Real-time Status**: Live monitoring of Ollama service and model availability
- **Interactive Examples**: Pre-built schemas and queries for different domains
- **SQL Validation**: Automatic syntax checking and error detection
- **Copy to Clipboard**: Easy result sharing and copying
- **Mobile Friendly**: Works perfectly on desktop, tablet, and mobile devices
- **Error Handling**: User-friendly error messages and troubleshooting
- **Multiple Model Support**: Works with CodeLlama, CodeBERT-base, and other models
- **Fast Training Integration**: Direct integration with --fast-mac training pipeline

## 🚀 Quick Start

### 1. Start the Web Application
```bash
# Easy startup (auto-installs dependencies) - default port 5000
python start_webapp.py

# With custom port
python start_webapp.py --port 8080

# With custom host and port
python start_webapp.py --host 127.0.0.1 --port 3000

# With debug mode and custom model
python start_webapp.py --port 8080 --debug --model-name my-sql-model

# Or run directly with custom options
pip install flask>=2.3.0 requests>=2.31.0
python app.py --port 8080 --host 0.0.0.0 --debug
```

**Available Options:**
- `--port, -p`: Port number (default: 5000)
- `--host`: Host address (default: 0.0.0.0)
- `--debug`: Enable debug mode
- `--model-name`: Ollama model name (default: text-to-sql)

### 2. Access the Interface
Open your browser and navigate to: **http://localhost:PORT** (where PORT is your chosen port, default 5000)

Examples:
- Default: http://localhost:5000
- Custom port: http://localhost:8080
- Custom host: http://127.0.0.1:3000

### 3. Use the Application
1. **Enter Database Schema**: Paste your CREATE TABLE statements
2. **Write Natural Query**: Describe what you want in plain English
3. **Generate SQL**: Click the generate button
4. **Copy Results**: Use the copy button to get the SQL code

## 📱 Interface Overview

### Main Page (`/`)
- **Schema Input**: Large text area for database schema
- **Query Input**: Text area for natural language query
- **Status Indicator**: Shows Ollama and model status
- **Generate Button**: Processes your request
- **Results Section**: Shows generated SQL with validation

### Examples Page (`/examples`)
- **Pre-built Examples**: 5 different complexity levels
- **Domain Coverage**: E-commerce, healthcare, inventory, etc.
- **One-click Loading**: Click "Use This Example" to load
- **Complexity Badges**: Basic, Intermediate, Advanced levels

## 🔧 API Endpoints

### Status Check
```bash
# Check CodeBERT model status
curl http://localhost:5000/api/status

# Response for CodeBERT setup
{
  "ollama_running": true,
  "model_available": true,
  "model_name": "codebert-sql",
  "model_type": "CodeBERT-base",
  "training_config": "fast-mac",
  "status": "ready"
}
```

### Generate SQL with CodeBERT
```bash
# CodeBERT model (fast response ~0.5-1s)
curl -X POST http://localhost:5000/api/generate-sql \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE users (id INT, name VARCHAR(50), age INT);",
    "query": "Find all users over 25"
  }'

# Expected CodeBERT response
{
  "sql": "SELECT * FROM users WHERE age > 25;",
  "status": "success",
  "model": "codebert-sql",
  "model_type": "CodeBERT-base",
  "response_time": 0.8,
  "training_mode": "fast-mac"
}

# Production model (higher quality ~1-2s)
curl -X POST http://localhost:8080/api/generate-sql \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2));",
    "query": "Show products under $50 ordered by price"
  }'
```

### Validate SQL
```bash
# Default port
curl -X POST http://localhost:5000/api/validate-sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM users;"}'

# Custom port  
curl -X POST http://localhost:8080/api/validate-sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM users;"}'
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the web app image
docker build -f Dockerfile.webapp -t text-to-sql-webapp .

# Run the container
docker run -p 5000:5000 text-to-sql-webapp
```

### Docker Compose (with Ollama)
```yaml
version: '3.8'
services:
  webapp:
    build:
      context: .
      dockerfile: Dockerfile.webapp
    ports:
      - "5000:5000"
    depends_on:
      - ollama
    environment:
      - OLLAMA_HOST=ollama:11434
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

## 🧪 Testing the Web App

### Automated Demo
```bash
# Run API demo with default port (requires web app to be running)
python demo_webapp.py

# Run API demo with custom port
python demo_webapp.py --port 8080

# Run API demo with custom host and port
python demo_webapp.py --host 127.0.0.1 --port 3000
```

### Manual Testing
1. Visit http://localhost:PORT (replace PORT with your chosen port)
2. Try the example schemas and queries
3. Check status indicators
4. Verify SQL generation and validation

## ⚙️ Configuration & Environment Variables

### CodeBERT Development Setup
```bash
# Environment variables for CodeBERT development
export OLLAMA_MODEL_NAME="codebert-sql"
export TRAINING_CONFIG="fast-mac"
export MODEL_TYPE="CodeBERT-base"
export FLASK_ENV="development"
export FLASK_DEBUG=1

# Start web app with CodeBERT configuration
python start_webapp.py --model-name codebert-sql --debug
```

### Production Environment Variables
```bash
# Production configuration
export OLLAMA_MODEL_NAME="text-to-sql"
export MODEL_TYPE="CodeLlama-7B"
export TRAINING_CONFIG="production"
export FLASK_ENV="production"
export FLASK_SECRET_KEY="your-secret-key-here"

# Start production web app
python start_webapp.py --port 80 --host 0.0.0.0
```

### Docker Environment (CodeBERT)
```yaml
# docker-compose.yml for CodeBERT development
version: '3.8'
services:
  webapp-codebert:
    build:
      context: .
      dockerfile: Dockerfile.webapp
    ports:
      - "5000:5000"
    environment:
      - OLLAMA_HOST=ollama:11434
      - OLLAMA_MODEL_NAME=codebert-sql
      - MODEL_TYPE=CodeBERT-base
      - TRAINING_CONFIG=fast-mac
      - FLASK_ENV=development
    depends_on:
      - ollama
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `FLASK_DEBUG`: Set to `False` for production
- `FLASK_PORT`: Default port number (can be overridden by --port)
- `FLASK_HOST`: Default host address (can be overridden by --host)
- `OLLAMA_MODEL_NAME`: Default model name (can be overridden by --model-name)
- `SECRET_KEY`: Change the secret key for production use

### Customization
- **Templates**: Modify HTML templates in `templates/` directory
- **Styling**: Update CSS in the template files
- **API Endpoints**: Add new routes in `app.py`

## 🚨 Troubleshooting

### Web App Won't Start
```bash
# Check if Flask is installed
pip list | grep Flask

# Install if missing
pip install flask>=2.3.0
```

### "Service Unavailable" Status
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Check if model exists
ollama list | grep text-to-sql

# Deploy model if missing
python deployment_script.py
```

### API Returns Errors
- Verify JSON format in requests
- Check request headers include `Content-Type: application/json`
- Ensure required fields (`schema`, `query`) are provided
- Check server logs for detailed error messages

### Token Generation Loops
If you see repeating `{ end }<|end|>` patterns:
```bash
# Run the fix script
python fix_ollama_tokens.py

# Or recreate the model
python deployment_script.py --force
```

### CodeBERT-Specific Issues

#### CodeBERT Model Not Found
```bash
# Check if CodeBERT model was deployed correctly
ollama list | grep codebert

# If missing, train and deploy CodeBERT
python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode full --model-name codebert-sql
```

#### Fast-Mac Training Issues
```bash
# If fast-mac training fails, check Mac MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Fallback to CPU-only training
python text_to_sql_train.py --cpu-mode --base-model microsoft/CodeBERT-base --mode train --max-samples 500
```

#### CodeBERT Response Quality
If CodeBERT generates poor SQL:
- **Expected**: Good for simple queries, may struggle with complex JOINs
- **Solution**: Use CodeBERT for development, switch to CodeLlama for production
- **Alternative**: Increase training samples: `--max-samples 2000`

#### Memory Issues with CodeBERT
```bash
# Monitor memory usage during training
python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train --max-samples 200

# Ultra-low memory mode
python text_to_sql_train.py --cpu-mode --base-model microsoft/CodeBERT-base --mode train --max-samples 100 --no-quantization
```

## 📊 Example Usage

### CodeBERT Development Examples

#### Basic Query (Good for CodeBERT)
```json
{
  "schema": "CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2));",
  "query": "Find all products under $50"
}
```

**CodeBERT Response:**
```json
{
  "sql": "SELECT * FROM products WHERE price < 50.00;",
  "model": "codebert-sql",
  "response_time": 0.6,
  "quality": "excellent"
}
```

#### Intermediate Query (Acceptable for CodeBERT)
```json
{
  "schema": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);",
  "query": "Show total sales for each month in 2024"
}
```

**CodeBERT Response:**
```json
{
  "sql": "SELECT DATE_FORMAT(order_date, '%Y-%m') as month, SUM(total) as total_sales FROM orders WHERE YEAR(order_date) = 2024 GROUP BY month;",
  "model": "codebert-sql",
  "response_time": 0.8,
  "quality": "good"
}
```

### Complex Query (Better with CodeLlama)
```json
{
  "schema": "CREATE TABLE customers (id INT, name VARCHAR(100)); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2)); CREATE TABLE order_items (id INT, order_id INT, product_id INT, quantity INT);",
  "query": "Show top 5 customers by total order value with their most purchased product"
}
```

**Recommendation**: Use CodeLlama-7B for complex multi-table queries like this.

## 📊 Model Performance Comparison

When choosing a model for your web app deployment:

| Model | Training Time | Memory Usage | SQL Quality | Web App Response | Best For |
|-------|---------------|--------------|-------------|------------------|----------|
| **CodeBERT-base** | 15-25 min | ~2GB | Good | Fast | Development/Testing |
| **CodeT5+ 770M** | 30-45 min | ~4GB | Very Good | Fast | Balanced Use |
| **CodeLlama-7B** | 2-3 hours | 8-16GB | Excellent | Medium | Production |
| **SQLCoder-7B** | 2-4 hours | 8-16GB | Excellent | Medium | Specialized SQL |

### Model Selection Guide

#### Choose **CodeBERT-base** for:
- 🚀 Rapid prototyping and testing
- 💻 Mac development environments
- 🔄 Frequent model updates
- 📚 Learning and experimentation
- ⚡ Quick demos and presentations

#### Choose **CodeLlama-7B** for:
- 🏢 Production deployments
- 🎯 Complex SQL requirements
- 💪 High-performance servers
- 📈 Large-scale applications

## 🚀 Model-Specific Quick Start

### CodeBERT-Base Model Setup (Recommended for Development)

For fast prototyping and testing, we recommend using Microsoft's CodeBERT-base model:

#### 1. Train CodeBERT Model
```bash
# Fast Mac training (15-25 minutes)
python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode train

# Full pipeline with deployment
python text_to_sql_train.py --fast-mac --base-model microsoft/CodeBERT-base --mode full --model-name codebert-sql
```

#### 2. Start Web App with CodeBERT
```bash
# Default setup with CodeBERT model
python start_webapp.py --model-name codebert-sql

# Custom port with CodeBERT
python start_webapp.py --port 8080 --model-name codebert-sql --debug
```

**CodeBERT Benefits for Web App:**
- ⚡ **Fast Training**: 15-25 minutes vs 2-3 hours
- 💾 **Low Memory**: ~2GB vs 8-16GB requirements
- 🍎 **Mac Optimized**: Perfect for Apple Silicon development
- 🔄 **Quick Iteration**: Rapid model updates and testing

### Production Model Setup (CodeLlama-7B)

For production deployments requiring highest SQL quality:

#### 1. Train Production Model
```bash
# Full training (2-3 hours)
python text_to_sql_train.py --mode full --max-samples 10000

# GPU-optimized training
python text_to_sql_train.py --mode full --device cuda --max-samples 20000
```

#### 2. Start Production Web App
```bash
# Production web app
python start_webapp.py --port 80 --host 0.0.0.0 --model-name text-to-sql
```
