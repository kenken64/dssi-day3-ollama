# Complete Text-to-SQL Model Training and Deployment Pipeline
# Author: Assistant
# Purpose: Convert Gretel AI dataset to Ollama text-to-SQL model

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

# Core ML libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_to_sql_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextToSQLConfig:
    """Configuration class for the text-to-SQL model training"""
    
    def __init__(self):
        # Model configuration
        self.base_model_name = "defog/sqlcoder-7b"  # Specialized SQL model
        self.model_output_dir = "./text-to-sql-model"
        self.final_model_dir = "./text-to-sql-final"
        
        # Training configuration
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.max_length = 1024
        self.validation_split = 0.1
        
        # LoRA configuration
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Quantization configuration
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = torch.float16
        self.bnb_4bit_quant_type = "nf4"
        
        # Dataset configuration
        self.dataset_name = "gretelai/synthetic_text_to_sql"
        self.max_samples = None  # Set to limit samples for testing
        
        # Ollama configuration
        self.ollama_model_name = "text-to-sql"
        self.modelfile_path = "./Modelfile"

class DataProcessor:
    """Process and prepare the Gretel AI dataset for training"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        
    def load_dataset(self) -> HFDataset:
        """Load the Gretel AI synthetic text-to-SQL dataset"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            dataset = load_dataset(self.config.dataset_name)
            logger.info(f"Dataset loaded successfully. Train size: {len(dataset['train'])}")
            
            # Limit samples if specified
            if self.config.max_samples:
                dataset['train'] = dataset['train'].select(range(self.config.max_samples))
                logger.info(f"Limited dataset to {self.config.max_samples} samples")
                
            return dataset['train']
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def format_sample(self, sample: Dict) -> Dict:
        """Format a single sample for instruction tuning"""
        
        # Create comprehensive prompt with context
        context = sample.get('sql_context', '')
        prompt = sample.get('sql_prompt', '')
        sql_query = sample.get('sql', '')
        explanation = sample.get('sql_explanation', '')
        domain = sample.get('domain', 'general')
        complexity = sample.get('sql_complexity', 'basic')
        
        # Enhanced instruction format
        instruction = f"""You are an expert SQL query generator for the {domain} domain. 
Generate an accurate SQL query based on the given database schema and natural language request.
Complexity level: {complexity}"""
        
        input_text = f"""Database Schema:
{context}

Request: {prompt}"""
        
        output_text = f"""```sql
{sql_query}
```

Explanation: {explanation}"""
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "domain": domain,
            "complexity": complexity
        }
    
    def prepare_dataset(self, dataset: HFDataset) -> Tuple[HFDataset, HFDataset]:
        """Prepare and split dataset for training"""
        logger.info("Formatting dataset samples...")
        
        # Format all samples
        formatted_data = []
        for sample in dataset:
            try:
                formatted_sample = self.format_sample(sample)
                formatted_data.append(formatted_sample)
            except Exception as e:
                logger.warning(f"Error formatting sample: {e}")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_data)} samples")
        
        # Split into train and validation
        train_data, val_data = train_test_split(
            formatted_data, 
            test_size=self.config.validation_split,
            random_state=42,
            stratify=[item['complexity'] for item in formatted_data]
        )
        
        # Convert to HuggingFace datasets
        train_dataset = HFDataset.from_list(train_data)
        val_dataset = HFDataset.from_list(val_data)
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def save_processed_data(self, train_dataset: HFDataset, val_dataset: HFDataset):
        """Save processed datasets"""
        os.makedirs("./processed_data", exist_ok=True)
        
        # Save as JSON lines
        with open("./processed_data/train.jsonl", "w") as f:
            for item in train_dataset:
                f.write(json.dumps(item) + "\n")
        
        with open("./processed_data/validation.jsonl", "w") as f:
            for item in val_dataset:
                f.write(json.dumps(item) + "\n")
        
        logger.info("Processed datasets saved to ./processed_data/")

class TextToSQLDataset(Dataset):
    """Custom dataset class for text-to-SQL training"""
    
    def __init__(self, dataset: HFDataset, tokenizer, max_length: int = 1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Combine instruction, input, and output
        full_text = f"{item['instruction']}\n\n{item['input']}\n\n{item['output']}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }

class ModelTrainer:
    """Handle model loading, training, and saving"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def setup_model_and_tokenizer(self):
        """Load and configure the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.base_model_name}")
        
        # Quantization configuration
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Configure model for training
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied")
    
    def train_model(self, train_dataset: HFDataset, val_dataset: HFDataset):
        """Train the model with the prepared dataset"""
        logger.info("Starting model training...")
        
        # Create custom datasets
        train_ds = TextToSQLDataset(train_dataset, self.tokenizer, self.config.max_length)
        val_ds = TextToSQLDataset(val_dataset, self.tokenizer, self.config.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.model_output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_drop_last=True,
            report_to="none"  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(self.config.final_model_dir)
        self.tokenizer.save_pretrained(self.config.final_model_dir)
        
        logger.info(f"Training completed. Model saved to {self.config.final_model_dir}")
        
        return trainer
    
    def evaluate_model(self, test_cases: List[Dict]) -> Dict:
        """Evaluate the trained model on test cases"""
        logger.info("Evaluating model...")
        
        if self.model is None:
            logger.error("Model not loaded. Please train or load a model first.")
            return {}
        
        results = []
        
        for i, case in enumerate(test_cases):
            try:
                prompt = f"Database Schema:\n{case['context']}\n\nRequest: {case['query']}"
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_sql = generated_text[len(prompt):].strip()
                
                results.append({
                    "test_case": i + 1,
                    "input_query": case['query'],
                    "expected": case.get('expected', 'N/A'),
                    "generated": generated_sql,
                    "context": case['context']
                })
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i + 1}: {e}")
                results.append({
                    "test_case": i + 1,
                    "error": str(e)
                })
        
        return results

class OllamaDeployer:
    """Handle Ollama model creation and deployment"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
    
    def create_modelfile(self):
        """Create Ollama Modelfile"""
        logger.info("Creating Ollama Modelfile...")
        
        modelfile_content = f"""FROM {self.config.base_model_name}

# Model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"

# System prompt for SQL generation
SYSTEM \"\"\"You are an expert SQL query generator. Given a natural language question and database schema context, generate accurate SQL queries.

Rules:
1. Always analyze the database schema carefully
2. Use proper SQL syntax and formatting
3. Include table aliases when joining multiple tables
4. Follow SQL best practices for performance
5. Provide clear explanations for complex queries
6. Handle edge cases and potential errors

Format your response as:
```sql
[YOUR SQL QUERY HERE]
```

Explanation: [Brief explanation of the query logic and any important considerations]
\"\"\"

# Template for conversations
TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
Database Schema:
{{ .Prompt }}

Generate the SQL query for the above request.<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
\"\"\"
"""
        
        with open(self.config.modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile created at {self.config.modelfile_path}")
    
    def deploy_to_ollama(self):
        """Deploy the model to Ollama"""
        logger.info("Deploying model to Ollama...")
        
        try:
            # Create Modelfile
            self.create_modelfile()
            
            # Create Ollama model
            import subprocess
            
            cmd = [
                "ollama", "create", 
                self.config.ollama_model_name, 
                "-f", self.config.modelfile_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully created Ollama model: {self.config.ollama_model_name}")
            else:
                logger.error(f"Error creating Ollama model: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error deploying to Ollama: {e}")
    
    def test_ollama_model(self):
        """Test the deployed Ollama model"""
        test_queries = [
            {
                "context": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100), created_at TIMESTAMP);",
                "query": "Find all users created in the last 30 days"
            },
            {
                "context": "CREATE TABLE orders (id INT, user_id INT, total DECIMAL(10,2), order_date DATE); CREATE TABLE users (id INT, name VARCHAR(50));",
                "query": "Get the total order amount for each user"
            }
        ]
        
        logger.info("Testing Ollama model...")
        
        try:
            import subprocess
            
            for i, test in enumerate(test_queries, 1):
                prompt = f"{test['context']}\n\nQuery: {test['query']}"
                
                cmd = ["ollama", "run", self.config.ollama_model_name, prompt]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                logger.info(f"Test {i} - Input: {test['query']}")
                logger.info(f"Test {i} - Output: {result.stdout}")
                
        except Exception as e:
            logger.error(f"Error testing Ollama model: {e}")

def create_test_cases() -> List[Dict]:
    """Create test cases for model evaluation"""
    return [
        {
            "context": "CREATE TABLE employees (id INT, name VARCHAR(50), salary DECIMAL(10,2), department VARCHAR(30), hire_date DATE);",
            "query": "Find all employees in the IT department with salary above 75000",
            "expected": "SELECT * FROM employees WHERE department = 'IT' AND salary > 75000;"
        },
        {
            "context": "CREATE TABLE sales (id INT, product_id INT, quantity INT, price DECIMAL(10,2), sale_date DATE);",
            "query": "Calculate total revenue by month for the last year",
            "expected": "SELECT DATE_FORMAT(sale_date, '%Y-%m') as month, SUM(quantity * price) as total_revenue FROM sales WHERE sale_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) GROUP BY month ORDER BY month;"
        },
        {
            "context": "CREATE TABLE customers (id INT, name VARCHAR(50), email VARCHAR(100)); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);",
            "query": "List customers with their total order amounts, including customers with no orders",
            "expected": "SELECT c.name, c.email, COALESCE(SUM(o.total), 0) as total_orders FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name, c.email;"
        }
    ]

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train and deploy text-to-SQL model")
    parser.add_argument("--mode", choices=["train", "deploy", "test", "full"], default="full", help="Execution mode")
    parser.add_argument("--max-samples", type=int, help="Limit number of training samples")
    parser.add_argument("--model-name", type=str, help="Custom model name for Ollama")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TextToSQLConfig()
    
    if args.max_samples:
        config.max_samples = args.max_samples
    
    if args.model_name:
        config.ollama_model_name = args.model_name
    
    logger.info(f"Starting text-to-SQL model pipeline in {args.mode} mode")
    
    try:
        if args.mode in ["train", "full"]:
            # Data processing
            processor = DataProcessor(config)
            dataset = processor.load_dataset()
            train_dataset, val_dataset = processor.prepare_dataset(dataset)
            processor.save_processed_data(train_dataset, val_dataset)
            
            # Model training
            trainer = ModelTrainer(config)
            trainer.setup_model_and_tokenizer()
            trainer.setup_lora()
            trainer.train_model(train_dataset, val_dataset)
            
            # Evaluation
            test_cases = create_test_cases()
            results = trainer.evaluate_model(test_cases)
            
            # Save evaluation results
            with open("evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info("Training completed successfully!")
        
        if args.mode in ["deploy", "full"]:
            # Ollama deployment
            deployer = OllamaDeployer(config)
            deployer.deploy_to_ollama()
            
        if args.mode in ["test", "full"]:
            # Test Ollama model
            deployer = OllamaDeployer(config)
            deployer.test_ollama_model()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
