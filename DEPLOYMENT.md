# Ollama Deployment Guide

## Prerequisites

1. **Install Ollama** (if not already installed):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

2. **Start Ollama service**:
```bash
ollama serve
```

## Deployment Steps

### Step 1: Ensure Your Model is Trained
Make sure you have a trained model in `./text-to-sql-final/` directory.

### Step 2: Quick Deployment (FIXED!)
```bash
# Deploy with default settings - now handles base model automatically
python text_to_sql_train.py --mode deploy

# Or with custom model name
python text_to_sql_train.py --mode deploy --model-name my-sql-assistant
```

**What's Fixed:**
- ✅ Automatically maps Hugging Face model names to Ollama names
- ✅ Downloads base model if not available (codellama:7b-instruct)
- ✅ Handles timeout and connection issues
- ✅ Provides fallback to alternative models

### Step 3: Test Your Deployment
```bash
# Test the deployed model
python text_to_sql_train.py --mode test

# Or manually test with Ollama
ollama run text-to-sql "Database Schema: CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100)); Request: Find all users with Gmail addresses"
```

## Model Name Mapping

The deployment now automatically maps training models to Ollama models:

| Training Model (Hugging Face) | Ollama Model |
|-------------------------------|--------------|
| `codellama/CodeLlama-7b-Instruct-hf` | `codellama:7b-instruct` |
| `codellama/CodeLlama-7b-hf` | `codellama:7b` |
| `defog/sqlcoder-7b` | `codellama:7b-instruct` |
| `Salesforce/codet5p-770m` | `codellama:7b-instruct` |
| Any other model | `codellama:7b-instruct` (fallback) |

## Alternative Manual Deployment

If you prefer manual control:

### 1. Create a Modelfile
```bash
# The deployment script automatically creates this, but you can customize:
cat > Modelfile << EOF
FROM codellama:7b-instruct

# Model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt for SQL generation
SYSTEM """You are an expert SQL query generator. Given a natural language question and database schema context, generate accurate SQL queries.

Rules:
1. Always analyze the database schema carefully
2. Use proper SQL syntax and formatting
3. Include table aliases when joining multiple tables
4. Follow SQL best practices for performance
5. Provide clear explanations for complex queries

Format your response as:
\`\`\`sql
[YOUR SQL QUERY HERE]
\`\`\`

Explanation: [Brief explanation of the query logic]
"""
EOF
```

### 2. Create the Ollama Model
```bash
# Create model from Modelfile
ollama create text-to-sql -f Modelfile

# Verify creation
ollama list
```

### 3. Test the Model
```bash
# Interactive test
ollama run text-to-sql

# Command line test
ollama run text-to-sql "Database Schema: CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50)); Request: Find all products under $50 in the electronics category"
```

## Troubleshooting

### Model Not Found
```bash
# Check available models
ollama list

# Recreate if missing
python text_to_sql_train.py --mode deploy --force
```

### Ollama Service Issues
```bash
# Check if Ollama is running
ollama ps

# Restart Ollama service
# macOS/Linux:
pkill ollama && ollama serve

# Windows: Restart from system tray or service manager
```

### Memory Issues
```bash
# For large models, ensure sufficient RAM
# Monitor with:
ollama ps

# If needed, use smaller model:
python text_to_sql_train.py --base-model Salesforce/codet5p-770m --mode full
```

## Usage Examples

### Basic Query
```bash
ollama run text-to-sql "Database Schema: CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(30), salary DECIMAL(10,2)); Request: Find all employees in Engineering with salary above 75000"
```

### Complex Query
```bash
ollama run text-to-sql "Database Schema: CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE); CREATE TABLE customers (id INT, name VARCHAR(50), city VARCHAR(30)); Request: Find the top 5 customers by total order value in the last 6 months"
```

### API Usage (Python)
```python
import subprocess
import json

def query_sql_model(schema, question):
    prompt = f"Database Schema: {schema}; Request: {question}"
    result = subprocess.run(
        ["ollama", "run", "text-to-sql", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout

# Example usage
schema = "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100));"
question = "Find all users with Gmail addresses"
sql_response = query_sql_model(schema, question)
print(sql_response)
```

## Configuration

You can customize the deployment by editing `config.yaml`:

```yaml
# Ollama Configuration
ollama:
  model_name: "text-to-sql"           # Change model name
  temperature: 0.1                    # Adjust creativity
  top_p: 0.9                         # Nucleus sampling
  top_k: 40                          # Top-k sampling
  repeat_penalty: 1.1                # Reduce repetition
```

## Performance Tips

1. **Use appropriate model size** for your hardware
2. **Limit context length** for faster responses
3. **Use batch processing** for multiple queries
4. **Monitor memory usage** with `ollama ps`
5. **Keep Ollama updated** for best performance
