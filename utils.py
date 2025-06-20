#!/usr/bin/env python3
"""
Utility functions for Text-to-SQL model training and deployment
"""

import os
import json
import yaml
import logging
import subprocess
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import torch
from transformers import AutoTokenizer
import sqlparse

class ConfigManager:
    """Manage configuration loading and validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_path} not found")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model": {
                "base_model_name": "defog/sqlcoder-7b",
                "model_output_dir": "./models/text-to-sql-checkpoint",
                "final_model_dir": "./models/text-to-sql-final"
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "max_length": 1024
            },
            "dataset": {
                "name": "gretelai/synthetic_text_to_sql",
                "max_samples": None
            },
            "ollama": {
                "model_name": "text-to-sql",
                "temperature": 0.1
            }
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save_config(self, output_path: str = None):
        """Save current configuration"""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

class SQLValidator:
    """Validate and format SQL queries"""
    
    @staticmethod
    def validate_sql(sql_query: str) -> tuple[bool, str]:
        """Validate SQL syntax"""
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Empty or invalid SQL"
            
            # Basic validation - check for common SQL keywords
            sql_upper = sql_query.upper().strip()
            valid_starters = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']
            
            if not any(sql_upper.startswith(starter) for starter in valid_starters):
                return False, "SQL does not start with a valid keyword"
            
            return True, "Valid SQL"
            
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"
    
    @staticmethod
    def format_sql(sql_query: str) -> str:
        """Format SQL query"""
        try:
            formatted = sqlparse.format(
                sql_query,
                reindent=True,
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=False
            )
            return formatted
        except Exception:
            return sql_query
    
    @staticmethod
    def extract_sql_from_text(text: str) -> Optional[str]:
        """Extract SQL from text that might contain markdown code blocks"""
        import re
        
        # Look for SQL in code blocks
        sql_pattern = r'```sql\s*(.*?)\s*```'
        match = re.search(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Look for SQL without code blocks
        sql_pattern = r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH).*?;'
        match = re.search(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(0).strip()
        
        return None

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.test_database = None
        self.setup_test_database()
    
    def setup_test_database(self):
        """Create a test SQLite database for query validation"""
        self.test_database = sqlite3.connect(":memory:")
        cursor = self.test_database.cursor()
        
        # Create sample tables
        cursor.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                salary REAL,
                hire_date DATE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                start_date DATE,
                end_date DATE,
                FOREIGN KEY (department_id) REFERENCES departments (id)
            )
        """)
        
        # Insert sample data
        sample_data = [
            ("INSERT INTO departments VALUES (1, 'Engineering', 1000000)", []),
            ("INSERT INTO departments VALUES (2, 'Marketing', 500000)", []),
            ("INSERT INTO employees VALUES (1, 'John Doe', 'Engineering', 75000, '2020-01-15')", []),
            ("INSERT INTO employees VALUES (2, 'Jane Smith', 'Marketing', 65000, '2019-03-20')", []),
            ("INSERT INTO projects VALUES (1, 'Web App', 1, '2023-01-01', '2023-12-31')", []),
        ]
        
        for query, params in sample_data:
            cursor.execute(query, params)
        
        self.test_database.commit()
    
    def test_query_execution(self, sql_query: str) -> tuple[bool, str, Any]:
        """Test if a SQL query can be executed"""
        try:
            cursor = self.test_database.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return True, "Query executed successfully", results
        
        except Exception as e:
            return False, f"Query execution failed: {str(e)}", None
    
    def evaluate_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Evaluate a list of SQL predictions"""
        results = {
            "total_queries": len(predictions),
            "syntactically_valid": 0,
            "executable": 0,
            "semantic_accuracy": 0,
            "errors": []
        }
        
        for i, pred in enumerate(predictions):
            sql_query = pred.get('generated_sql', '')
            expected_sql = pred.get('expected_sql', '')
            
            # Check syntax validity
            is_valid, validation_msg = SQLValidator.validate_sql(sql_query)
            if is_valid:
                results["syntactically_valid"] += 1
            
            # Check executability
            can_execute, exec_msg, _ = self.test_query_execution(sql_query)
            if can_execute:
                results["executable"] += 1
            
            # Store errors for analysis
            if not is_valid or not can_execute:
                results["errors"].append({
                    "query_index": i,
                    "sql": sql_query,
                    "validation_error": validation_msg if not is_valid else None,
                    "execution_error": exec_msg if not can_execute else None
                })
        
        # Calculate percentages
        total = results["total_queries"]
        if total > 0:
            results["syntax_accuracy"] = results["syntactically_valid"] / total
            results["execution_accuracy"] = results["executable"] / total
        
        return results

class OllamaManager:
    """Manage Ollama operations"""
    
    def __init__(self, model_name: str = "text-to-sql"):
        self.model_name = model_name
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service"""
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            logging.error("Ollama not found. Please install Ollama first.")
            return False
    
    def model_exists(self) -> bool:
        """Check if the model exists in Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True
            )
            return self.model_name in result.stdout
        except subprocess.SubprocessError:
            return False
    
    def create_model(self, modelfile_path: str) -> bool:
        """Create Ollama model from Modelfile"""
        try:
            result = subprocess.run(
                ["ollama", "create", self.model_name, "-f", modelfile_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully created Ollama model: {self.model_name}")
                return True
            else:
                logging.error(f"Failed to create model: {result.stderr}")
                return False
                
        except subprocess.SubprocessError as e:
            logging.error(f"Error creating Ollama model: {e}")
            return False
    
    def query_model(self, prompt: str, timeout: int = 60) -> Optional[str]:
        """Query the Ollama model"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logging.error(f"Query failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logging.error("Query timed out")
            return None
        except subprocess.SubprocessError as e:
            logging.error(f"Error querying model: {e}")
            return None
    
    def query_model_safe(self, prompt: str, timeout: int = 60, max_retries: int = 3) -> Optional[str]:
        """Query the Ollama model with better error handling and token management"""
        for attempt in range(max_retries):
            try:
                # Use a more structured prompt format
                formatted_prompt = f"""Database Schema:
{prompt.split('Request:')[0].strip()}

Query Request: {prompt.split('Request:')[1].strip() if 'Request:' in prompt else prompt}

Generate a SQL query for the above request. Respond with only the SQL code in a code block followed by a brief explanation."""
                
                result = subprocess.run(
                    ["ollama", "run", self.model_name],
                    input=formatted_prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    
                    # Check for token loop issue
                    if "{ end }<|end|>" in response or response.count("<|end|>") > 5:
                        logging.warning(f"Detected token loop in response, retrying... (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            time.sleep(1)  # Brief pause before retry
                            continue
                        else:
                            logging.error("Model stuck in token loop, trying fallback")
                            return self._fallback_query(prompt)
                    
                    return response
                else:
                    logging.error(f"Query failed: {result.stderr}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None
                    
            except subprocess.TimeoutExpired:
                logging.error(f"Query timed out (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    continue
                return None
            except subprocess.SubprocessError as e:
                logging.error(f"Error querying model: {e}")
                if attempt < max_retries - 1:
                    continue
                return None
        
        return None
    
    def _fallback_query(self, prompt: str) -> str:
        """Fallback query method when model gets stuck"""
        # Simple fallback response
        if "SELECT" in prompt.upper() or "find" in prompt.lower() or "get" in prompt.lower():
            return """```sql
SELECT * FROM table_name WHERE condition;
```
Note: The model encountered an issue. Please provide a more specific schema and query for better results."""
        return "Error: Unable to generate SQL query. Please check your input and try again."

class DatasetAnalyzer:
    """Analyze the text-to-SQL dataset"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
    
    def analyze_complexity_distribution(self, dataset) -> Dict[str, int]:
        """Analyze SQL complexity distribution"""
        complexity_counts = {}
        
        for sample in dataset:
            complexity = sample.get('sql_complexity', 'unknown')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return complexity_counts
    
    def analyze_domain_distribution(self, dataset) -> Dict[str, int]:
        """Analyze domain distribution"""
        domain_counts = {}
        
        for sample in dataset:
            domain = sample.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return domain_counts
    
    def analyze_query_length_distribution(self, dataset) -> Dict[str, float]:
        """Analyze SQL query length distribution"""
        lengths = []
        
        for sample in dataset:
            sql_query = sample.get('sql', '')
            lengths.append(len(sql_query.split()))
        
        if lengths:
            return {
                'mean_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'median_length': sorted(lengths)[len(lengths) // 2]
            }
        
        return {}
    
    def generate_analysis_report(self, dataset) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        return {
            'total_samples': len(dataset),
            'complexity_distribution': self.analyze_complexity_distribution(dataset),
            'domain_distribution': self.analyze_domain_distribution(dataset),
            'query_length_stats': self.analyze_query_length_distribution(dataset),
            'timestamp': pd.Timestamp.now().isoformat()
        }

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements"""
    requirements = {
        'torch_available': False,
        'cuda_available': False,
        'transformers_available': False,
        'ollama_available': False,
        'sufficient_memory': False
    }
    
    try:
        import torch
        requirements['torch_available'] = True
        requirements['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import transformers
        requirements['transformers_available'] = True
    except ImportError:
        pass
    
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True)
        requirements['ollama_available'] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Check available RAM (simplified)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        requirements['sufficient_memory'] = available_gb > 8  # 8GB minimum
    except ImportError:
        requirements['sufficient_memory'] = True  # Assume sufficient if can't check
    
    return requirements

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "models",
        "models/text-to-sql-checkpoint", 
        "models/text-to-sql-final",
        "processed_data",
        "logs",
        "cache",
        "outputs",
        "configs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logging.info("Directory structure created successfully")

if __name__ == "__main__":
    # Example usage
    setup_logging()
    
    # Check system requirements
    reqs = check_system_requirements()
    print("System Requirements Check:")
    for req, status in reqs.items():
        print(f"  {req}: {'✓' if status else '✗'}")
    
    # Create directory structure
    create_directory_structure()
    
    # Load configuration
    config_manager = ConfigManager()
    print(f"Base model: {config_manager.get('model.base_model_name')}")
    print(f"Batch size: {config_manager.get('training.batch_size')}")
