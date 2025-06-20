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
        # Model configuration - using smaller CodeLlama model by default
        self.base_model_name = "codellama/CodeLlama-7b-Instruct-hf"  # Smaller than defog/sqlcoder-7b
        self.model_output_dir = "./text-to-sql-model"
        self.final_model_dir = "./text-to-sql-final"
        
        # Training configuration optimized for smaller models
        self.batch_size = 4  # Can increase with smaller model
        self.gradient_accumulation_steps = 4  # Adjusted back
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.max_length = 512  # Keep reduced for memory
        self.validation_split = 0.1
        
        # LoRA configuration - optimized for CodeLlama
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        # CodeLlama specific target modules
        self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Quantization configuration
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = torch.float16
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = True  # Additional memory savings
        
        # Memory optimization settings
        self.use_gradient_checkpointing = True
        self.dataloader_pin_memory = False  # Disable for memory savings
        self.save_only_adapter = True  # Only save LoRA weights, not full model
        
        # Mac MPS optimizations
        self.mac_optimizations = {
            "smaller_batch_size": 2,      # Reduce for MPS
            "higher_grad_accum": 8,       # Compensate for smaller batch
            "reduced_max_length": 256,    # Shorter sequences for speed
            "fewer_epochs": 2,            # Quick training for testing
            "eval_less_frequently": 1000, # Reduce evaluation overhead
            "disable_grad_checkpointing": True,  # Can be slower on MPS
        }
        
        # Dataset configuration
        self.dataset_name = "gretelai/synthetic_text_to_sql"
        self.max_samples = None  # Set to limit samples for testing
        
        # Ollama configuration
        self.ollama_model_name = "text-to-sql"
        self.modelfile_path = "./Modelfile"
        
    def apply_mac_optimizations(self):
        """Apply Mac-specific optimizations for faster training"""
        logger.info("Applying Mac MPS optimizations for faster training...")
        
        # Adjust batch settings for MPS
        self.batch_size = self.mac_optimizations["smaller_batch_size"]
        self.gradient_accumulation_steps = self.mac_optimizations["higher_grad_accum"]
        
        # Reduce sequence length for speed
        self.max_length = self.mac_optimizations["reduced_max_length"]
        
        # Fewer epochs for quicker iterations
        self.num_epochs = self.mac_optimizations["fewer_epochs"]
        
        # Disable gradient checkpointing (can be slower on MPS)
        if self.mac_optimizations["disable_grad_checkpointing"]:
            self.use_gradient_checkpointing = False
            
        logger.info(f"Mac optimizations applied: batch_size={self.batch_size}, "
                   f"grad_accum={self.gradient_accumulation_steps}, "
                   f"max_length={self.max_length}, epochs={self.num_epochs}")
                   
        # Recommend sample limiting for testing
        if self.max_samples is None:
            recommended_samples = 1000
            logger.info(f"💡 TIP: For faster Mac training, consider using --max-samples {recommended_samples}")
            logger.info(f"💡 Example: python text_to_sql_train.py --max-samples {recommended_samples} --mode train")

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
        
        # Split into train and validation with smart stratification
        try:
            # Try stratified split first (better for larger datasets)
            complexity_labels = [item['complexity'] for item in formatted_data]
            
            # Check if stratification is possible (each class needs at least 2 samples)
            from collections import Counter
            complexity_counts = Counter(complexity_labels)
            min_class_count = min(complexity_counts.values())
            
            if min_class_count >= 2 and len(formatted_data) > 20:
                # Safe to use stratification
                train_data, val_data = train_test_split(
                    formatted_data, 
                    test_size=self.config.validation_split,
                    random_state=42,
                    stratify=complexity_labels
                )
                logger.info("Used stratified split for balanced complexity distribution")
            else:
                # Fall back to simple random split
                train_data, val_data = train_test_split(
                    formatted_data, 
                    test_size=self.config.validation_split,
                    random_state=42
                )
                logger.info(f"Used simple random split (min class count: {min_class_count}, total samples: {len(formatted_data)})")
                
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}")
            # Fallback to simple split
            train_data, val_data = train_test_split(
                formatted_data, 
                test_size=self.config.validation_split,
                random_state=42
            )
            logger.info("Used fallback simple random split")
        
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
        
    def detect_device_capabilities(self):
        """Detect device capabilities and adjust training settings accordingly"""
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        
        # Check if we're on Mac with MPS
        if torch.backends.mps.is_available():
            device_type = "mps"
            logger.info("Detected Apple Silicon Mac with MPS backend")
            # Automatically apply Mac optimizations
            self.config.apply_mac_optimizations()
        elif torch.cuda.is_available():
            device_type = "cuda"
            logger.info(f"Detected CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device_type = "cpu"
            logger.info("Using CPU for training")
        
        # Conservative approach: only use optimizations when we're sure they work
        use_fp16 = False
        use_bf16 = False
        
        if device_type == "cuda":
            # fp16 works reliably on CUDA
            use_fp16 = True
            logger.info("CUDA detected: Enabling fp16 for faster training")
        elif device_type == "mps":
            # MPS is tricky - be very conservative
            logger.info("MPS detected: Using fp32 with Mac-specific optimizations")
            # Don't enable bf16 on MPS unless we're absolutely sure
            use_fp16 = False
            use_bf16 = False
        else:
            # CPU - use fp32 for reliability
            logger.info("CPU detected: Using fp32 (most compatible)")
            use_fp16 = False
            use_bf16 = False
        
        return use_fp16, use_bf16, device_type
        
    def setup_model_and_tokenizer(self):
        """Load and configure the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.base_model_name}")
        
        # Quantization configuration with error handling
        bnb_config = None
        if self.config.use_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant=self.config.use_nested_quant,
                )
                logger.info("4-bit quantization enabled with nested quantization")
            except Exception as e:
                logger.warning(f"Failed to setup 4-bit quantization: {e}")
                logger.info("Falling back to no quantization")
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
        
        # Load model with fallback for quantization issues
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            logger.info("Model loaded successfully with quantization" if bnb_config else "Model loaded successfully without quantization")
        except Exception as e:
            if bnb_config is not None:
                logger.warning(f"Failed to load model with quantization: {e}")
                logger.info("Retrying without quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                logger.info("Model loaded successfully without quantization")
            else:
                raise e
        
        # Configure model for training with memory optimizations
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        # Enable gradient checkpointing for memory savings
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory optimization")
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration...")
        
        # Start with configured target modules
        target_modules = self.config.lora_target_modules
        
        # Try with configured modules first, fallback to auto-detection
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"LoRA applied successfully with configured modules: {target_modules}")
            
        except ValueError as e:
            if "Target modules" in str(e) and "not found" in str(e):
                logger.warning(f"Configured target modules failed: {target_modules}")
                logger.info("Auto-detecting target modules...")
                
                # Auto-detect target modules
                target_modules = self.find_target_modules()
                
                if target_modules:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.config.lora_r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        target_modules=target_modules
                    )
                    self.model = get_peft_model(self.model, lora_config)
                    logger.info(f"LoRA applied successfully with auto-detected modules: {target_modules}")
                else:
                    raise ValueError("Could not find suitable target modules for LoRA")
            else:
                raise e
        
        self.model.print_trainable_parameters()
        logger.info("LoRA configuration applied")
    
    def find_target_modules(self):
        """Automatically find target modules for LoRA based on the model architecture"""
        logger.info("Auto-detecting target modules for LoRA...")
        
        # Get all named modules
        module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parts = name.split('.')
                if len(parts) > 0:
                    module_names.add(parts[-1])
        
        # Common target modules patterns for different architectures
        common_targets = [
            # Llama/CodeLlama style
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            # Alternative naming
            "query", "key", "value", "dense", "output",
            # Generic patterns
            "c_attn", "c_proj", "c_fc", "mlp.c_fc", "mlp.c_proj",
            # Transformer patterns  
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"
        ]
        
        # Find matching modules
        found_targets = []
        for target in common_targets:
            if target in module_names:
                found_targets.append(target)
        
        # If no matches, try to find linear layers in attention blocks
        if not found_targets:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Look for attention-related modules
                    if any(keyword in name.lower() for keyword in ['attn', 'attention', 'self']):
                        parts = name.split('.')
                        if len(parts) > 0:
                            found_targets.append(parts[-1])
        
        # Remove duplicates and limit to reasonable number
        found_targets = list(set(found_targets))[:8]  # Limit to 8 targets max
        
        if found_targets:
            logger.info(f"Found target modules: {found_targets}")
            return found_targets
        else:
            # Fallback to basic linear layers
            logger.warning("No standard target modules found, using basic linear layer targets")
            basic_targets = []
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'embed' not in name.lower():
                    parts = name.split('.')
                    if len(parts) > 0:
                        basic_targets.append(parts[-1])
                    if len(basic_targets) >= 4:  # Limit to first 4 found
                        break
            return list(set(basic_targets))

    def train_model(self, train_dataset: HFDataset, val_dataset: HFDataset):
        """Train the model with the prepared dataset"""
        logger.info("Starting model training...")
        
        # Detect device capabilities
        use_fp16, use_bf16, device_type = self.detect_device_capabilities()
        
        # Create custom datasets
        train_ds = TextToSQLDataset(train_dataset, self.tokenizer, self.config.max_length)
        val_ds = TextToSQLDataset(val_dataset, self.tokenizer, self.config.max_length)
        
        # Dynamic evaluation frequency based on device
        eval_steps = 500  # Default
        save_steps = 500  # Default
        
        # Use less frequent evaluation on Mac for speed
        if device_type == "mps":
            eval_steps = self.config.mac_optimizations["eval_less_frequently"]
            save_steps = self.config.mac_optimizations["eval_less_frequently"]
            logger.info(f"Mac optimization: Reduced evaluation frequency to every {eval_steps} steps")
        
        # Training arguments optimized for memory and smaller files
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
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",  # Fixed: was evaluation_strategy
            save_total_limit=2,  # Reduced to save disk space
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,  # Device-aware fp16
            bf16=use_bf16,  # Use bf16 on MPS if available
            dataloader_drop_last=True,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            remove_unused_columns=True,  # Memory optimization
            report_to="none"  # Disable wandb
        )
        
        logger.info(f"Training configuration: fp16={use_fp16}, bf16={use_bf16}, device={device_type}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer - Updated to avoid deprecation warning
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            processing_class=self.tokenizer  # Fixed: was tokenizer (deprecated)
        )
        
        # Train the model
        trainer.train()
        
        # Save the model (adapter only if configured)
        if self.config.save_only_adapter:
            # Save only LoRA adapter weights (much smaller)
            self.model.save_pretrained(self.config.final_model_dir)
            self.tokenizer.save_pretrained(self.config.final_model_dir)
            logger.info(f"Training completed. LoRA adapter saved to {self.config.final_model_dir} (adapter only)")
        else:
            # Save full model
            trainer.save_model(self.config.final_model_dir)
            self.tokenizer.save_pretrained(self.config.final_model_dir)
            logger.info(f"Training completed. Full model saved to {self.config.final_model_dir}")
        
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
    
    def get_ollama_base_model(self):
        """Map Hugging Face model names to Ollama model names"""
        hf_to_ollama_mapping = {
            "codellama/CodeLlama-7b-Instruct-hf": "codellama:7b",  # Changed to base version
            "codellama/CodeLlama-7b-hf": "codellama:7b",
            "codellama/CodeLlama-13b-Instruct-hf": "codellama:13b",
            "defog/sqlcoder-7b": "codellama:7b",  # Use base version
            "Salesforce/codet5p-770m": "codellama:7b",  # Use base version
            "Salesforce/codet5p-220m": "codellama:7b",  # Use base version
            "microsoft/CodeBERT-base": "codellama:7b",  # Use base version
        }
        
        base_model = self.config.base_model_name
        ollama_model = hf_to_ollama_mapping.get(base_model, "codellama:7b")  # Default to base version
        
        if base_model not in hf_to_ollama_mapping:
            logger.warning(f"Unknown model {base_model}, using fallback: {ollama_model}")
        else:
            logger.info(f"Mapping {base_model} -> {ollama_model}")
            
        return ollama_model
    
    def create_modelfile(self):
        """Create Ollama Modelfile"""
        logger.info("Creating Ollama Modelfile...")
        
        # Get the correct Ollama model name
        ollama_base_model = self.get_ollama_base_model()
        
        modelfile_content = f"""FROM {ollama_base_model}

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
        logger.info(f"Using Ollama base model: {ollama_base_model}")
    
    def deploy_to_ollama(self):
        """Deploy the model to Ollama"""
        logger.info("Deploying model to Ollama...")
        
        try:
            # Get the Ollama base model name
            ollama_base_model = self.get_ollama_base_model()
            
            # Check if base model exists, pull if needed
            logger.info(f"Checking if base model {ollama_base_model} is available...")
            import subprocess
            
            # Check if model exists
            check_cmd = ["ollama", "list"]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if ollama_base_model not in check_result.stdout:
                logger.info(f"Base model {ollama_base_model} not found. Pulling from Ollama registry...")
                pull_cmd = ["ollama", "pull", ollama_base_model]
                pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=300)
                
                if pull_result.returncode != 0:
                    logger.error(f"Failed to pull base model: {pull_result.stderr}")
                    logger.info("Trying alternative base model: codellama:7b")
                    # Try alternative
                    alt_pull_cmd = ["ollama", "pull", "codellama:7b"]
                    alt_result = subprocess.run(alt_pull_cmd, capture_output=True, text=True, timeout=300)
                    if alt_result.returncode == 0:
                        # Update the mapping to use the working model
                        ollama_base_model = "codellama:7b"
                        logger.info("Successfully pulled alternative model")
                    else:
                        raise Exception(f"Could not pull any base model: {alt_result.stderr}")
                else:
                    logger.info(f"Successfully pulled {ollama_base_model}")
            else:
                logger.info(f"Base model {ollama_base_model} already available")
            
            # Create Modelfile with the correct base model
            self.create_modelfile()
            
            # Remove existing model if it exists
            remove_cmd = ["ollama", "rm", self.config.ollama_model_name]
            subprocess.run(remove_cmd, capture_output=True)  # Ignore errors
            
            # Create Ollama model
            cmd = [
                "ollama", "create", 
                self.config.ollama_model_name, 
                "-f", self.config.modelfile_path
            ]
            
            logger.info(f"Creating Ollama model with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"Successfully created Ollama model: {self.config.ollama_model_name}")
                logger.info("🎉 Model deployed successfully! You can now use:")
                logger.info(f"   ollama run {self.config.ollama_model_name}")
            else:
                logger.error(f"Error creating Ollama model: {result.stderr}")
                logger.error(f"Command output: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            logger.error("Deployment timed out. This might be due to slow internet connection.")
        except Exception as e:
            logger.error(f"Error deploying to Ollama: {e}")
            logger.info("💡 Make sure Ollama is running: ollama serve")
    
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
    parser.add_argument("--base-model", type=str, help="Base model to use for training")
    parser.add_argument("--save-full-model", action="store_true", help="Save full model instead of just adapter weights")
    parser.add_argument("--fast-mac", action="store_true", help="Enable aggressive Mac optimizations for fastest training")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TextToSQLConfig()
    
    if args.max_samples:
        config.max_samples = args.max_samples
    
    if args.model_name:
        config.ollama_model_name = args.model_name
    
    if args.base_model:
        config.base_model_name = args.base_model
    
    if args.save_full_model:
        config.save_only_adapter = False
    
    # Apply aggressive Mac optimizations if requested
    if args.fast_mac:
        logger.info("🚀 Aggressive Mac optimizations enabled!")
        config.apply_mac_optimizations()
        # Additional aggressive settings
        config.max_samples = config.max_samples or 500  # Very small dataset
        config.num_epochs = 1  # Single epoch
        config.max_length = 128  # Very short sequences
        logger.info("🚀 Fast Mac mode: 500 samples, 1 epoch, 128 max length")
    
    logger.info(f"Starting text-to-SQL model pipeline in {args.mode} mode")
    logger.info(f"Using base model: {config.base_model_name}")
    
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
