# Complete Text-to-SQL Model Training and Deployment Pipeline
# Purpose: Convert Gretel AI dataset to Ollama text-to-SQL model
# Description: This script provides a comprehensive pipeline for training text-to-SQL models
#              using the Gretel AI synthetic dataset, with support for multiple devices
#              (CPU, Mac MPS, NVIDIA GPU) and efficient deployment to Ollama

# Standard library imports for file operations, JSON handling, logging, and command-line arguments
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

# Core PyTorch and machine learning libraries
import torch  # Main PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network modules and functions
from torch.utils.data import Dataset, DataLoader  # Data loading utilities

# Hugging Face Transformers library for pre-trained models and training
from transformers import (
    AutoModelForCausalLM,  # Automatic model loading for causal language models
    AutoTokenizer,  # Automatic tokenizer loading
    TrainingArguments,  # Configuration for training parameters
    Trainer,  # High-level training API
    DataCollatorForLanguageModeling,  # Data collation for language modeling
    BitsAndBytesConfig  # Configuration for quantization
)

# Hugging Face Datasets library for dataset loading and processing
from datasets import load_dataset, Dataset as HFDataset

# PEFT (Parameter-Efficient Fine-Tuning) library for LoRA
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Additional utilities for evaluation and data splitting
import evaluate
from sklearn.model_selection import train_test_split

# Setup comprehensive logging to track training progress and debug issues
# Logs are written to both a file and console for monitoring and troubleshooting
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO for detailed progress tracking
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp and level
    handlers=[
        logging.FileHandler('text_to_sql_training.log'),  # Write logs to file for persistence
        logging.StreamHandler()  # Also display logs in console for real-time monitoring
    ]
)
logger = logging.getLogger(__name__)  # Create logger instance for this module

class TextToSQLConfig:
    """
    Configuration class for the text-to-SQL model training pipeline.
    
    This class centralizes all configuration parameters for training, including:
    - Model and dataset settings
    - Training hyperparameters optimized for different hardware
    - Device-specific optimizations (CPU, Mac MPS, NVIDIA GPU)
    - LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning
    - Memory optimization settings for resource-constrained environments
    """
    
    def __init__(self):
        # Model configuration - using smaller CodeLlama model by default for better compatibility
        # CodeLlama-7b-Instruct is specifically designed for code generation tasks
        self.base_model_name = "codellama/CodeLlama-7b-Instruct-hf"  # Smaller than defog/sqlcoder-7b
        self.model_output_dir = "./text-to-sql-model"  # Temporary training outputs
        self.final_model_dir = "./text-to-sql-final"   # Final model location
        
        # Training configuration optimized for smaller models and memory efficiency
        # These parameters balance training speed, memory usage, and model quality
        self.batch_size = 4  # Number of samples processed together (higher = faster but more memory)
        self.gradient_accumulation_steps = 4  # Accumulate gradients to simulate larger batch sizes
        self.learning_rate = 2e-4  # Learning rate optimized for LoRA fine-tuning
        self.num_epochs = 3  # Number of complete passes through the dataset
        self.max_length = 512  # Maximum sequence length (reduced for memory efficiency)
        self.validation_split = 0.1  # Fraction of data used for validation (10%)
        
        # LoRA (Low-Rank Adaptation) configuration - enables efficient fine-tuning
        # LoRA adds small adapter layers instead of updating the entire model
        self.lora_r = 16  # Rank of the adaptation matrices (higher = more parameters)
        self.lora_alpha = 32  # Scaling factor for LoRA weights (typically 2x the rank)
        self.lora_dropout = 0.1  # Dropout rate for LoRA layers to prevent overfitting
        # CodeLlama specific target modules - these are the layers where LoRA adapters are applied
        self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Quantization configuration for memory optimization
        # 4-bit quantization reduces model size by ~75% with minimal quality loss
        self.use_4bit = True  # Enable 4-bit quantization using bitsandbytes
        self.bnb_4bit_compute_dtype = torch.float16  # Compute type for quantized operations
        self.bnb_4bit_quant_type = "nf4"  # Quantization type (nf4 is optimal for most models)
        self.use_nested_quant = True  # Additional memory savings through nested quantization
        
        # Memory optimization settings for resource-constrained environments
        self.use_gradient_checkpointing = True  # Trade computation for memory (slower but less RAM)
        self.dataloader_pin_memory = False  # Disable pinned memory to save RAM
        self.save_only_adapter = True  # Only save LoRA weights, not the full model (much smaller files)
        
        # Mac MPS (Metal Performance Shaders) optimizations for Apple Silicon
        # MPS provides GPU acceleration on Mac but has different characteristics than CUDA
        self.mac_optimizations = {
            "smaller_batch_size": 2,      # Reduce batch size as MPS has limited memory bandwidth
            "higher_grad_accum": 8,       # Compensate for smaller batches with more accumulation
            "reduced_max_length": 256,    # Shorter sequences for faster MPS processing
            "fewer_epochs": 2,            # Quick training iterations for testing/development
            "eval_less_frequently": 1000, # Reduce evaluation overhead on MPS
            "disable_grad_checkpointing": True,  # Gradient checkpointing can be slower on MPS
        }
        
        # NVIDIA GPU optimizations for maximum performance on CUDA devices
        # CUDA GPUs have high memory bandwidth and excellent optimization support
        self.nvidia_optimizations = {
            "larger_batch_size": 8,       # Leverage GPU memory for larger batches
            "use_bf16": True,             # Brain float 16 for better precision than fp16
            "enable_grad_checkpointing": True,  # Save memory for larger models
            "faster_evaluation": 500,     # More frequent evaluation since GPU can handle it
            "use_flash_attention": True,  # Advanced attention optimization (if available)
        }
        
        # CPU optimizations for systems without dedicated GPU acceleration
        # CPU training is slower but works on any system
        self.cpu_optimizations = {
            "smaller_batch_size": 1,      # Very small batches due to limited CPU memory
            "higher_grad_accum": 16,      # High accumulation to compensate for tiny batches
            "reduced_max_length": 256,    # Reduce computation complexity
            "fewer_epochs": 1,            # Single epoch for reasonable training time
            "disable_quantization": True, # Quantization may not work well on all CPUs
            "eval_less_frequently": 2000, # Minimize evaluation overhead
        }
        
        # Dataset configuration for Gretel AI synthetic text-to-SQL dataset
        self.dataset_name = "gretelai/synthetic_text_to_sql"  # High-quality synthetic SQL dataset
        self.max_samples = None  # Set to limit samples for testing (None = use full dataset)
        
        # Ollama deployment configuration
        self.ollama_model_name = "text-to-sql"  # Name for the deployed Ollama model
        self.modelfile_path = "./Modelfile"     # Path to Ollama model configuration file
        
    def apply_mac_optimizations(self):
        """
        Apply Mac-specific optimizations for faster training on Apple Silicon.
        
        Mac MPS (Metal Performance Shaders) provides GPU acceleration but has different
        characteristics than CUDA. These optimizations account for MPS limitations
        and optimize for typical Mac development workflows.
        """
        logger.info("Applying Mac MPS optimizations for faster training...")
        
        # Adjust batch settings for MPS memory characteristics
        self.batch_size = self.mac_optimizations["smaller_batch_size"]
        self.gradient_accumulation_steps = self.mac_optimizations["higher_grad_accum"]
        
        # Reduce sequence length for faster MPS processing
        self.max_length = self.mac_optimizations["reduced_max_length"]
        
        # Fewer epochs for quicker development iterations
        self.num_epochs = self.mac_optimizations["fewer_epochs"]
        
        # Disable gradient checkpointing as it can be slower on MPS
        if self.mac_optimizations["disable_grad_checkpointing"]:
            self.use_gradient_checkpointing = False
            
        logger.info(f"Mac optimizations applied: batch_size={self.batch_size}, "
                   f"grad_accum={self.gradient_accumulation_steps}, "
                   f"max_length={self.max_length}, epochs={self.num_epochs}")
                   
        # Provide helpful recommendations for Mac users
        if self.max_samples is None:
            recommended_samples = 1000
            logger.info(f"💡 TIP: For faster Mac training, consider using --max-samples {recommended_samples}")
            logger.info(f"💡 Example: python text_to_sql_train.py --max-samples {recommended_samples} --mode train")

    def apply_nvidia_optimizations(self):
        """
        Apply NVIDIA GPU-specific optimizations for maximum performance.
        
        NVIDIA CUDA GPUs have excellent optimization support and high memory bandwidth.
        These settings leverage GPU capabilities for faster training with larger models.
        """
        logger.info("Applying NVIDIA GPU optimizations for maximum performance...")
        
        # Leverage GPU memory and compute power with larger batches
        self.batch_size = self.nvidia_optimizations["larger_batch_size"]
        self.gradient_accumulation_steps = max(1, self.gradient_accumulation_steps // 2)
        
        # Enable gradient checkpointing for memory efficiency with large models
        if self.nvidia_optimizations["enable_grad_checkpointing"]:
            self.use_gradient_checkpointing = True
            
        # More frequent evaluation since GPU can handle the computational overhead
        eval_steps = self.nvidia_optimizations["faster_evaluation"]
        
        logger.info(f"NVIDIA optimizations applied: batch_size={self.batch_size}, "
                   f"grad_accum={self.gradient_accumulation_steps}, "
                   f"eval_steps={eval_steps}, checkpointing={self.use_gradient_checkpointing}")
                   
        # GPU users can benefit from full dataset training
        if self.max_samples is None:
            logger.info("💡 TIP: GPU detected - you can train on the full dataset for best results")
        
    def apply_cpu_optimizations(self):
        """
        Apply CPU-specific optimizations for efficient training without GPU.
        
        CPU training is slower but works on any system. These optimizations minimize
        training time while ensuring compatibility across different CPU architectures.
        """
        logger.info("Applying CPU optimizations for efficient training...")
        
        # CPU requires very conservative settings due to limited memory and compute
        self.batch_size = self.cpu_optimizations["smaller_batch_size"]
        self.gradient_accumulation_steps = self.cpu_optimizations["higher_grad_accum"]
        
        # Reduce sequence length for faster CPU computation
        self.max_length = self.cpu_optimizations["reduced_max_length"]
        
        # Single epoch for reasonable training time on CPU
        self.num_epochs = self.cpu_optimizations["fewer_epochs"]
        
        # Disable quantization if it causes compatibility issues on CPU
        if self.cpu_optimizations["disable_quantization"]:
            self.use_4bit = False
            logger.info("Disabled 4-bit quantization for CPU compatibility")
            
        logger.info(f"CPU optimizations applied: batch_size={self.batch_size}, "
                   f"grad_accum={self.gradient_accumulation_steps}, "
                   f"max_length={self.max_length}, epochs={self.num_epochs}")
                   
        # Strong recommendation for reasonable CPU training time
        if self.max_samples is None:
            recommended_samples = 500
            logger.info(f"💡 TIP: For reasonable CPU training time, use --max-samples {recommended_samples}")

class DataProcessor:
    """
    Process and prepare the Gretel AI dataset for training.
    
    This class handles:
    - Loading the Gretel AI synthetic text-to-SQL dataset from Hugging Face
    - Formatting samples into instruction-tuning format for the model
    - Creating train/validation splits with intelligent stratification
    - Saving processed data for reproducibility and debugging
    """
    
    def __init__(self, config: TextToSQLConfig):
        """Initialize the data processor with configuration settings."""
        self.config = config
        
    def load_dataset(self) -> HFDataset:
        """
        Load the Gretel AI synthetic text-to-SQL dataset from Hugging Face.
        
        The Gretel AI dataset contains high-quality synthetic examples of:
        - Natural language questions
        - Corresponding database schemas
        - SQL queries that answer the questions
        - Domain information and complexity levels
        
        Returns:
            HFDataset: The loaded dataset ready for processing
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            # Load dataset from Hugging Face hub
            dataset = load_dataset(self.config.dataset_name)
            logger.info(f"Dataset loaded successfully. Train size: {len(dataset['train'])}")
            
            # Limit samples if specified for testing or resource constraints
            if self.config.max_samples:
                dataset['train'] = dataset['train'].select(range(self.config.max_samples))
                logger.info(f"Limited dataset to {self.config.max_samples} samples")
                
            return dataset['train']
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def format_sample(self, sample: Dict) -> Dict:
        """
        Format a single sample for instruction tuning.
        
        Converts raw dataset samples into a structured format suitable for training:
        - Creates clear instructions for the model
        - Formats database schema and user request as input
        - Structures the expected SQL output with explanations
        - Preserves metadata like domain and complexity for analysis
        
        Args:
            sample: Raw sample from the Gretel AI dataset containing context, prompt, SQL, etc.
            
        Returns:
            Dict: Formatted sample with instruction, input, output, and metadata
        """
        
        # Extract all relevant fields from the raw sample
        context = sample.get('sql_context', '')        # Database schema/table definitions
        prompt = sample.get('sql_prompt', '')          # Natural language question
        sql_query = sample.get('sql', '')              # Expected SQL query
        explanation = sample.get('sql_explanation', '') # Human explanation of the query
        domain = sample.get('domain', 'general')       # Domain (e.g., finance, healthcare)
        complexity = sample.get('sql_complexity', 'basic') # Complexity level
        
        # Create comprehensive instruction that sets clear expectations for the model
        instruction = f"""You are an expert SQL query generator for the {domain} domain. 
Generate an accurate SQL query based on the given database schema and natural language request.
Complexity level: {complexity}"""
        
        # Format the input with clear structure for the model to understand
        input_text = f"""Database Schema:
{context}

Request: {prompt}"""
        
        # Format the expected output with SQL code blocks and explanation
        output_text = f"""```sql
{sql_query}
```

Explanation: {explanation}"""
        
        return {
            "instruction": instruction,  # What the model should do
            "input": input_text,        # What the model receives
            "output": output_text,      # What the model should generate
            "domain": domain,           # Metadata for analysis
            "complexity": complexity    # Metadata for stratification
        }
    
    def prepare_dataset(self, dataset: HFDataset) -> Tuple[HFDataset, HFDataset]:
        """
        Prepare and split dataset for training with intelligent stratification.
        
        This method:
        1. Formats all dataset samples for instruction tuning
        2. Attempts stratified splitting to ensure balanced complexity distribution
        3. Falls back to simple random splitting for small datasets
        4. Creates train/validation splits suitable for model evaluation
        
        Args:
            dataset: Raw Hugging Face dataset
            
        Returns:
            Tuple of (train_dataset, validation_dataset) as Hugging Face datasets
        """
        logger.info("Formatting dataset samples...")
        
        # Format all samples for instruction tuning, skipping any that fail
        formatted_data = []
        for sample in dataset:
            try:
                formatted_sample = self.format_sample(sample)
                formatted_data.append(formatted_sample)
            except Exception as e:
                logger.warning(f"Error formatting sample: {e}")
                continue  # Skip problematic samples rather than failing entirely
        
        logger.info(f"Successfully formatted {len(formatted_data)} samples")
        
        # Intelligent data splitting with fallback strategies
        try:
            # Extract complexity labels for stratified splitting
            # Stratification ensures balanced representation of different complexity levels
            complexity_labels = [item['complexity'] for item in formatted_data]
            
            # Check if stratification is mathematically possible
            # Each class needs at least 2 samples for stratified splitting
            from collections import Counter
            complexity_counts = Counter(complexity_labels)
            min_class_count = min(complexity_counts.values())
            
            if min_class_count >= 2 and len(formatted_data) > 20:
                # Use stratified split for balanced complexity distribution
                train_data, val_data = train_test_split(
                    formatted_data, 
                    test_size=self.config.validation_split,
                    random_state=42,  # Fixed seed for reproducibility
                    stratify=complexity_labels
                )
                logger.info("Used stratified split for balanced complexity distribution")
            else:
                # Fall back to simple random split for small datasets
                train_data, val_data = train_test_split(
                    formatted_data, 
                    test_size=self.config.validation_split,
                    random_state=42
                )
                logger.info(f"Used simple random split (min class count: {min_class_count}, total samples: {len(formatted_data)})")
                
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}")
            # Final fallback to simple split if stratification fails for any reason
            train_data, val_data = train_test_split(
                formatted_data, 
                test_size=self.config.validation_split,
                random_state=42
            )
            logger.info("Used fallback simple random split")
        
        # Convert Python lists to Hugging Face Dataset objects for compatibility with Transformers
        train_dataset = HFDataset.from_list(train_data)
        val_dataset = HFDataset.from_list(val_data)
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def save_processed_data(self, train_dataset: HFDataset, val_dataset: HFDataset):
        """
        Save processed datasets to disk for reproducibility and debugging.
        
        Saves datasets in JSON Lines format, which is:
        - Human-readable for inspection
        - Easy to load for debugging
        - Compact and efficient
        
        Args:
            train_dataset: Processed training dataset
            val_dataset: Processed validation dataset
        """
        # Create directory for processed data
        os.makedirs("./processed_data", exist_ok=True)
        
        # Save training data as JSON Lines (one JSON object per line)
        with open("./processed_data/train.jsonl", "w") as f:
            for item in train_dataset:
                f.write(json.dumps(item) + "\n")
        
        # Save validation data as JSON Lines
        with open("./processed_data/validation.jsonl", "w") as f:
            for item in val_dataset:
                f.write(json.dumps(item) + "\n")
        
        logger.info("Processed datasets saved to ./processed_data/")

class TextToSQLDataset(Dataset):
    """
    Custom PyTorch dataset class for text-to-SQL training.
    
    This class handles:
    - Tokenization of instruction-formatted text for language modeling
    - Proper padding and truncation for consistent batch sizes
    - Label creation for causal language modeling (next-token prediction)
    
    The dataset formats each sample as a complete text sequence that the model
    learns to generate, following the instruction-tuning paradigm.
    """
    
    def __init__(self, dataset: HFDataset, tokenizer, max_length: int = 1024):
        """
        Initialize the dataset with tokenizer and length constraints.
        
        Args:
            dataset: Hugging Face dataset containing formatted samples
            tokenizer: Pre-trained tokenizer for the model
            max_length: Maximum sequence length (longer sequences are truncated)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single tokenized sample for training.
        
        Combines instruction, input, and output into a single text sequence
        that the model learns to generate through causal language modeling.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict containing input_ids, attention_mask, and labels for training
        """
        item = self.dataset[idx]
        
        # Combine all parts into a single training sequence
        # The model learns to generate the entire sequence, with the output
        # being the "ground truth" it should produce
        full_text = f"{item['instruction']}\n\n{item['input']}\n\n{item['output']}"
        
        # Tokenize the complete sequence with proper padding and truncation
        encoding = self.tokenizer(
            full_text,
            truncation=True,              # Cut off at max_length if too long
            max_length=self.max_length,   # Consistent sequence lengths
            padding="max_length",         # Pad short sequences to max_length
            return_tensors="pt"           # Return PyTorch tensors
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),      # Token IDs for the model
            "attention_mask": encoding["attention_mask"].flatten(), # Mask for padding tokens
            "labels": encoding["input_ids"].flatten()          # Labels = input_ids for causal LM
        }

class ModelTrainer:
    """
    Handle model loading, training, and saving with device-aware optimizations.
    
    This class provides:
    - Automatic device detection and optimization (CPU, Mac MPS, NVIDIA GPU)
    - Model loading with quantization and fallback strategies
    - LoRA (Low-Rank Adaptation) setup for efficient fine-tuning
    - Training with memory optimizations and device-specific settings
    - Model evaluation and saving in various formats
    """
    
    def __init__(self, config: TextToSQLConfig):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config: Configuration object containing all training parameters
        """
        self.config = config
        self.model = None      # Will hold the loaded model
        self.tokenizer = None  # Will hold the tokenizer
        
    def detect_device_capabilities(self):
        """
        Detect device capabilities and adjust training settings accordingly.
        
        This method:
        1. Detects available hardware (CPU, Mac MPS, NVIDIA GPU)
        2. Automatically applies device-specific optimizations
        3. Determines the best precision settings (fp16, bf16, fp32)
        4. Provides helpful information about detected hardware
        
        Returns:
            Tuple of (use_fp16, use_bf16, device_type) for training configuration
        """
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        
        # Detect device type and apply corresponding optimizations
        if torch.backends.mps.is_available():
            device_type = "mps"
            logger.info("Detected Apple Silicon Mac with MPS backend")
            # Automatically apply Mac optimizations for better MPS performance
            self.config.apply_mac_optimizations()
        elif torch.cuda.is_available():
            device_type = "cuda"
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Detected NVIDIA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            # Automatically apply NVIDIA optimizations for maximum performance
            self.config.apply_nvidia_optimizations()
        else:
            device_type = "cpu"
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1e9
            logger.info(f"Using CPU for training: {cpu_count} cores, {memory_gb:.1f}GB RAM")
            # Automatically apply CPU optimizations for efficiency
            self.config.apply_cpu_optimizations()
        
        # Determine optimal precision settings based on device capabilities
        use_fp16 = False  # 16-bit floating point for speed
        use_bf16 = False  # Brain float 16 for better numerical stability
        
        if device_type == "cuda":
            # Check GPU compute capability for advanced precision support
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:  # Modern GPUs (A100, RTX 30/40 series)
                use_bf16 = True
                logger.info("GPU supports bf16: Enabling bf16 for optimal training")
            elif major >= 7:  # Older GPUs (V100, RTX 20 series, GTX 16 series)
                use_fp16 = True
                logger.info("GPU supports fp16: Enabling fp16 for faster training")
            else:
                logger.info("Older GPU detected: Using fp32 for compatibility")
        elif device_type == "mps":
            # MPS (Apple Silicon) is tricky - be conservative for stability
            logger.info("MPS detected: Using fp32 with Mac-specific optimizations")
            # Test MPS fp16 support carefully (known to have issues)
            try:
                test_tensor = torch.randn(1, 1, device='mps', dtype=torch.float16)
                del test_tensor
                # Keep fp16 disabled for now due to MPS stability issues
                logger.info("MPS fp16 test passed, but staying with fp32 for stability")
            except Exception as e:
                logger.info(f"MPS fp16 test failed: {e}")
            use_fp16 = False
            use_bf16 = False
        else:
            # CPU - use fp32 for maximum compatibility and reliability
            logger.info("CPU detected: Using fp32 (most compatible)")
            use_fp16 = False
            use_bf16 = False
        
        return use_fp16, use_bf16, device_type
        
    def setup_model_and_tokenizer(self, device_type="auto"):
        """
        Load and configure the base model and tokenizer with device-aware optimizations.
        
        This method handles:
        1. Device-specific quantization configuration
        2. Tokenizer loading with padding token setup
        3. Model loading with multiple fallback strategies
        4. Memory optimization settings
        
        Args:
            device_type: Target device type ("auto", "cpu", "cuda", "mps")
        """
        logger.info(f"Loading model: {self.config.base_model_name}")
        
        # Configure quantization based on device capabilities
        bnb_config = None
        if self.config.use_4bit and device_type != "cpu":
            try:
                # Adjust quantization compute type based on device characteristics
                compute_dtype = self.config.bnb_4bit_compute_dtype
                if device_type == "mps":
                    # MPS may have compatibility issues with certain data types
                    compute_dtype = torch.float32
                    logger.info("Using float32 for MPS quantization compatibility")
                elif device_type == "cuda":
                    # CUDA can handle float16/bfloat16 efficiently
                    compute_dtype = self.config.bnb_4bit_compute_dtype
                
                # Create quantization configuration for memory efficiency
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,                                    # Enable 4-bit quantization
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type, # Use NF4 quantization
                    bnb_4bit_compute_dtype=compute_dtype,                 # Compute precision
                    bnb_4bit_use_double_quant=self.config.use_nested_quant, # Additional compression
                )
                logger.info(f"4-bit quantization enabled for {device_type} with {compute_dtype}")
            except Exception as e:
                logger.warning(f"Failed to setup 4-bit quantization: {e}")
                logger.info("Falling back to no quantization")
                bnb_config = None
        elif device_type == "cpu":
            logger.info("CPU detected: Quantization disabled for compatibility")
            bnb_config = None
        
        # Load tokenizer with proper configuration
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True  # Allow custom tokenizer code if needed
        )
        
        # Ensure tokenizer has a padding token (required for batch processing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with device-aware configuration and comprehensive fallbacks
        try:
            # Configure device-specific settings for optimal performance
            torch_dtype = torch.float16  # Default precision
            device_map = "auto"          # Default device mapping
            
            if device_type == "mps":
                # MPS (Apple Silicon) works better with specific settings
                torch_dtype = torch.float32  # MPS has issues with fp16 in some cases
                device_map = None            # Let MPS handle device placement automatically
                logger.info("Using float32 and MPS device mapping")
            elif device_type == "cpu":
                torch_dtype = torch.float32  # CPU requires float32 for stability
                device_map = None            # CPU doesn't need device mapping
                logger.info("Using float32 for CPU")
            elif device_type == "cuda":
                # CUDA can handle float16 efficiently
                torch_dtype = torch.float16
                device_map = "auto"          # Let Transformers handle GPU memory allocation
                logger.info("Using float16 and auto device mapping for CUDA")
            
            # Attempt model loading with optimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=bnb_config,   # Apply quantization if configured
                device_map=device_map,            # Device placement strategy
                trust_remote_code=True,           # Allow custom model code
                torch_dtype=torch_dtype,          # Precision for model weights
                low_cpu_mem_usage=True if device_type != "cpu" else False  # Memory optimization
            )
            logger.info(f"Model loaded successfully on {device_type}" + 
                       (" with quantization" if bnb_config else " without quantization"))
                       
        except Exception as e:
            if bnb_config is not None:
                # First fallback: Retry without quantization if it caused issues
                logger.warning(f"Failed to load model with quantization: {e}")
                logger.info("Retrying without quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True if device_type != "cpu" else False
                )
                logger.info(f"Model loaded successfully on {device_type} without quantization")
            else:
                # Final fallback: Use most compatible settings possible
                logger.warning(f"Standard loading failed: {e}")
                logger.info("Trying most compatible settings...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32  # Most compatible precision
                )
                logger.info("Model loaded with fallback settings")
        
        # Configure model for training with memory optimizations
        self.model.config.use_cache = False      # Disable KV cache for training
        self.model.config.pretraining_tp = 1     # Tensor parallelism setting
        
        # Enable gradient checkpointing for memory savings (trades compute for memory)
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
    """
    Create comprehensive test cases for model evaluation.
    
    These test cases cover:
    - Basic SELECT queries with WHERE conditions
    - Complex aggregations with GROUP BY and time functions
    - Multi-table JOINs with different join types
    - Various SQL complexity levels and domains
    
    Returns:
        List of test case dictionaries with context, query, and expected SQL
    """
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
    """
    Main execution function that orchestrates the entire training and deployment pipeline.
    
    This function:
    1. Parses command-line arguments for flexible execution
    2. Initializes configuration with user preferences
    3. Applies device-specific optimizations based on arguments
    4. Executes the requested pipeline mode (train, deploy, test, or full)
    5. Handles errors gracefully with informative logging
    
    The pipeline supports multiple execution modes:
    - train: Only train the model
    - deploy: Only deploy to Ollama
    - test: Only test the deployed model
    - full: Complete pipeline (train -> deploy -> test)
    """
    # Set up comprehensive command-line argument parsing
    parser = argparse.ArgumentParser(description="Train and deploy text-to-SQL model")
    
    # Core execution parameters
    parser.add_argument("--mode", choices=["train", "deploy", "test", "full"], default="full", 
                        help="Execution mode: train only, deploy only, test only, or full pipeline")
    parser.add_argument("--max-samples", type=int, 
                        help="Limit number of training samples (useful for testing/development)")
    parser.add_argument("--model-name", type=str, 
                        help="Custom model name for Ollama deployment")
    parser.add_argument("--base-model", type=str, 
                        help="Base model to use for training (overrides default)")
    
    # Training configuration options
    parser.add_argument("--save-full-model", action="store_true", 
                        help="Save full model instead of just LoRA adapter weights (larger file)")
    
    # Device and optimization options
    parser.add_argument("--fast-mac", action="store_true", 
                        help="Enable aggressive Mac optimizations for fastest possible training")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", 
                        help="Force specific device (auto=detect automatically)")
    parser.add_argument("--no-quantization", action="store_true", 
                        help="Disable 4-bit quantization (uses more memory)")
    parser.add_argument("--cpu-mode", action="store_true", 
                        help="Force CPU-only training with CPU-specific optimizations")

    args = parser.parse_args()
    
    # Initialize configuration with default settings
    config = TextToSQLConfig()
    
    # Apply user-specified configuration overrides
    if args.max_samples:
        config.max_samples = args.max_samples
    
    if args.model_name:
        config.ollama_model_name = args.model_name
    
    if args.base_model:
        config.base_model_name = args.base_model
    
    if args.save_full_model:
        config.save_only_adapter = False
    
    # Apply aggressive Mac optimizations if requested (development/testing mode)
    if args.fast_mac:
        logger.info("🚀 Aggressive Mac optimizations enabled!")
        config.apply_mac_optimizations()
        # Additional ultra-fast settings for quick iterations
        config.max_samples = config.max_samples or 500  # Very small dataset
        config.num_epochs = 1                           # Single epoch only
        config.max_length = 128                         # Very short sequences
        logger.info("🚀 Fast Mac mode: 500 samples, 1 epoch, 128 max length")
    
    # Handle device-specific configuration
    if args.cpu_mode or args.device == "cpu":
        logger.info("🖥️  CPU-only mode enabled")
        config.apply_cpu_optimizations()
        config.use_4bit = False  # Disable quantization for CPU compatibility
        
    if args.no_quantization:
        logger.info("🔧 Quantization disabled by user")
        config.use_4bit = False
        
    # Override automatic device detection if user specified a device
    forced_device = None
    if args.device != "auto":
        forced_device = args.device
        logger.info(f"🎯 Device forced to: {forced_device}")

    # Log pipeline configuration
    logger.info(f"Starting text-to-SQL model pipeline in {args.mode} mode")
    logger.info(f"Using base model: {config.base_model_name}")
    
    try:
        # Execute training phase if requested
        if args.mode in ["train", "full"]:
            logger.info("=" * 50)
            logger.info("TRAINING PHASE")
            logger.info("=" * 50)
            
            # Data processing and preparation
            processor = DataProcessor(config)
            dataset = processor.load_dataset()
            train_dataset, val_dataset = processor.prepare_dataset(dataset)
            processor.save_processed_data(train_dataset, val_dataset)
            
            # Model training with device detection/forcing
            trainer = ModelTrainer(config)
            
            # Handle device configuration (forced vs. automatic detection)
            if forced_device:
                # Use user-specified device and apply appropriate optimizations
                device_type = forced_device
                if device_type == "cpu":
                    config.apply_cpu_optimizations()
                elif device_type == "mps":
                    config.apply_mac_optimizations()
                elif device_type == "cuda":
                    config.apply_nvidia_optimizations()
                use_fp16, use_bf16 = False, False  # Will be determined by device type
                if device_type == "cuda":
                    use_fp16 = True
                logger.info(f"Using forced device: {device_type}")
            else:
                # Automatic device detection and optimization
                use_fp16, use_bf16, device_type = trainer.detect_device_capabilities()
            
            # Load model and apply LoRA for efficient fine-tuning
            trainer.setup_model_and_tokenizer(device_type)
            trainer.setup_lora()
            trainer.train_model(train_dataset, val_dataset)
            
            # Evaluate model performance on test cases
            test_cases = create_test_cases()
            results = trainer.evaluate_model(test_cases)
            
            # Save evaluation results for analysis
            with open("evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info("✅ Training completed successfully!")
        
        # Execute deployment phase if requested
        if args.mode in ["deploy", "full"]:
            logger.info("=" * 50)
            logger.info("DEPLOYMENT PHASE")
            logger.info("=" * 50)
            
            # Deploy trained model to Ollama
            deployer = OllamaDeployer(config)
            deployer.deploy_to_ollama()
            
        # Execute testing phase if requested
        if args.mode in ["test", "full"]:
            logger.info("=" * 50)
            logger.info("TESTING PHASE")
            logger.info("=" * 50)
            
            # Test deployed Ollama model
            deployer = OllamaDeployer(config)
            deployer.test_ollama_model()
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    """
    Script entry point - executes main() when script is run directly.
    
    This allows the script to be imported as a module without automatically
    executing the training pipeline, while still providing command-line functionality
    when run directly with: python text_to_sql_train.py [arguments]
    
    Example usage:
    - python text_to_sql_train.py --mode full --fast-mac
    - python text_to_sql_train.py --mode train --max-samples 1000 --device cuda
    - python text_to_sql_train.py --mode deploy --model-name my-sql-model
    """
    main()
