# Creating an Ollama Text-to-SQL Model from Gretel AI Dataset

## Overview
This guide walks you through creating a custom Ollama model for text-to-SQL conversion using the `gretelai/synthetic_text_to_sql` dataset from Hugging Face.

## Dataset Analysis
The dataset contains:
- **Size**: 100K+ examples
- **Domains**: 25+ domains (cybersecurity, healthcare, finance, etc.)
- **SQL Complexity**: Basic SQL, aggregations, joins, subqueries, window functions
- **Format**: Natural language prompts → SQL queries + explanations

## Step 1: Environment Setup

```bash
# Install required packages
pip install datasets transformers torch accelerate bitsandbytes
pip install ollama

# Start Ollama service
ollama serve
```

## Step 2: Download and Prepare Dataset

```python
from datasets import load_dataset
import json
import os

# Load the dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql")

# Examine the dataset structure
print("Dataset keys:", dataset.keys())
print("Train size:", len(dataset['train']))
print("Sample entry:", dataset['train'][0])

# Format data for training
def format_sample(sample):
    """Format each sample for instruction tuning"""
    return {
        "instruction": "Convert the following natural language query to SQL:",
        "input": f"Context: {sample['sql_context']}\n\nQuery: {sample['sql_prompt']}",
        "output": f"```sql\n{sample['sql']}\n```\n\nExplanation: {sample['sql_explanation']}"
    }

# Apply formatting
formatted_dataset = dataset['train'].map(format_sample)

# Save as JSONL for training
def save_as_jsonl(dataset, filename):
    with open(filename, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

save_as_jsonl(formatted_dataset, 'text_to_sql_training.jsonl')
```

## Step 3: Choose Base Model and Fine-tuning Approach

### Option A: Fine-tune with Hugging Face Transformers

```python
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# Choose a base model (examples)
MODEL_OPTIONS = [
    "microsoft/DialoGPT-medium",
    "microsoft/CodeBERT-base",
    "codellama/CodeLlama-7b-Instruct-hf",
    "defog/sqlcoder-7b"  # Specialized for SQL
]

model_name = "defog/sqlcoder-7b"  # Recommended for SQL tasks

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # For memory efficiency
    device_map="auto"
)

# Add padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
```

### Training Configuration

```python
# Training arguments
training_args = TrainingArguments(
    output_dir="./text-to-sql-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=3,
    remove_unused_columns=False,
)

# Custom data collator
def data_collator(batch):
    # Combine instruction, input, and output
    texts = []
    for item in batch:
        text = f"{item['instruction']}\n{item['input']}\n{item['output']}"
        texts.append(text)
    
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Labels are the same as input_ids for causal LM
    encoded['labels'] = encoded['input_ids'].clone()
    
    return encoded

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./text-to-sql-final")
```

## Step 4: Convert to Ollama Format

### Option A: Create Modelfile

```dockerfile
# Create a Modelfile
FROM codellama:7b

# Set custom parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

# System prompt for SQL generation
SYSTEM """You are an expert SQL query generator. Given a natural language question and database schema context, generate accurate SQL queries.

Rules:
1. Always analyze the schema carefully
2. Use proper SQL syntax
3. Include table aliases when needed
4. Provide explanations for complex queries
5. Follow best practices for performance

Format your response as:
```sql
[YOUR SQL QUERY HERE]
```

Explanation: [Brief explanation of the query logic]
"""

# Template for conversations
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
```

### Create the Ollama Model

```bash
# Create the model
ollama create text-to-sql -f Modelfile

# Test the model
ollama run text-to-sql "Given a users table with columns id, name, email, created_at, generate SQL to find all users created in the last 30 days"
```

### Option B: Import Fine-tuned Model

```python
# Convert your fine-tuned model to GGUF format (if needed)
# This requires additional tools like llama.cpp

# Save model in compatible format
model.save_pretrained("./ollama-ready-model")
tokenizer.save_pretrained("./ollama-ready-model")
```

## Step 5: Advanced Configuration

### Custom Training Data Preparation

```python
# Enhanced formatting with domain-specific examples
def enhanced_format_sample(sample):
    domain_context = f"Domain: {sample['domain']}"
    complexity_note = f"Complexity: {sample['sql_complexity']}"
    
    return {
        "instruction": f"Generate SQL query for the {sample['domain']} domain.",
        "input": f"{domain_context}\n{complexity_note}\n\nContext: {sample['sql_context']}\n\nQuery: {sample['sql_prompt']}",
        "output": f"```sql\n{sample['sql']}\n```\n\nExplanation: {sample['sql_explanation']}"
    }

# Filter by complexity for progressive training
basic_data = dataset['train'].filter(lambda x: x['sql_complexity'] == 'basic SQL')
advanced_data = dataset['train'].filter(lambda x: x['sql_complexity'] in ['subqueries', 'window functions'])
```

### Model Validation

```python
# Test queries for validation
test_cases = [
    {
        "context": "CREATE TABLE users (id INT, name VARCHAR(50), age INT);",
        "query": "Find all users older than 25",
        "expected": "SELECT * FROM users WHERE age > 25;"
    },
    {
        "context": "CREATE TABLE orders (id INT, user_id INT, total DECIMAL(10,2), order_date DATE);",
        "query": "Get total sales by month",
        "expected": "SELECT DATE_FORMAT(order_date, '%Y-%m') as month, SUM(total) FROM orders GROUP BY month;"
    }
]

def validate_model(model, test_cases):
    for case in test_cases:
        prompt = f"Context: {case['context']}\nQuery: {case['query']}"
        # Generate response and compare
        pass
```

## Step 6: Usage Examples

```bash
# Start your custom model
ollama run text-to-sql

# Example queries:
# 1. Basic query
ollama run text-to-sql "Context: CREATE TABLE employees (id INT, name VARCHAR(50), salary DECIMAL(10,2), department VARCHAR(30)); Query: Find all employees in the IT department with salary above 50000"

# 2. Complex aggregation
ollama run text-to-sql "Context: CREATE TABLE sales (id INT, product_id INT, quantity INT, sale_date DATE); Query: Show monthly sales trends for the last year"

# 3. Join query
ollama run text-to-sql "Context: CREATE TABLE customers (id INT, name VARCHAR(50)); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2)); Query: List customers with their total order amounts"
```

## Best Practices

1. **Start Small**: Begin with basic SQL examples and gradually increase complexity
2. **Domain Specialization**: Train separate models for specific domains if needed
3. **Validation**: Always test generated queries on sample databases
4. **Version Control**: Keep track of different model versions
5. **Performance Monitoring**: Monitor query accuracy and execution time

## Troubleshooting

### Common Issues:
- **Memory errors**: Use smaller batch sizes or gradient checkpointing
- **Convergence issues**: Adjust learning rate and training epochs
- **Poor SQL quality**: Increase training data or improve prompt engineering

### Performance Optimization:
- Use quantization for smaller models
- Implement caching for common queries
- Add query validation layers

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [SQL Best Practices](https://www.sqlstyle.guide/)

This guide provides a complete pipeline from dataset preparation to deployment of your custom text-to-SQL model using Ollama!