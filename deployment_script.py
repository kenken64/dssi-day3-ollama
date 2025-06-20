#!/usr/bin/env python3
"""
Deployment script for Text-to-SQL Ollama model
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils import (
    ConfigManager, 
    OllamaManager, 
    ModelEvaluator, 
    SQLValidator,
    setup_logging,
    check_system_requirements
)

class ModelDeployer:
    """Handle model deployment to Ollama"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.ollama_manager = OllamaManager(
            self.config.get('ollama.model_name', 'text-to-sql')
        )
        setup_logging(
            self.config.get('logging.level', 'INFO'),
            self.config.get('logging.log_file', 'deployment.log')
        )
        
    def create_modelfile(self) -> str:
        """Create Ollama Modelfile"""
        modelfile_path = self.config.get('ollama.modelfile_path', './Modelfile')
        base_model = self.config.get('model.base_model_name', 'codellama:7b')
        
        # Extract model name for Ollama (remove organization prefix if present)
        if '/' in base_model:
            ollama_base = base_model.split('/')[-1]
        else:
            ollama_base = base_model
            
        modelfile_content = f"""FROM {ollama_base}

# Model parameters for SQL generation
PARAMETER temperature {self.config.get('ollama.temperature', 0.1)}
PARAMETER top_p {self.config.get('ollama.top_p', 0.9)}
PARAMETER top_k {self.config.get('ollama.top_k', 40)}
PARAMETER repeat_penalty {self.config.get('ollama.repeat_penalty', 1.1)}
PARAMETER num_predict 512
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "<|end|>"
PARAMETER stop "{{ end }}"
PARAMETER stop "END"
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"

# System prompt optimized for SQL generation
SYSTEM \"\"\"You are an expert SQL query generator. Convert natural language requests into accurate SQL queries based on the provided database schema.

Guidelines:
1. Analyze the database schema carefully
2. Generate syntactically correct SQL
3. Use proper formatting and table aliases
4. Be concise and accurate

Response format: Return only the SQL query in a code block, followed by a brief explanation.

Example:
```sql
SELECT * FROM users WHERE created_at > DATE_SUB(NOW(), INTERVAL 30 DAY);
```
This query finds all users created in the last 30 days.\"\"\"

# Simple template to avoid token confusion
TEMPLATE \"\"\"{{ .System }}

Database Schema:
{{ .Prompt }}

Please generate the SQL query:\"\"\"
"""
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logging.info(f"Modelfile created at {modelfile_path}")
        return modelfile_path
    
    def deploy_model(self, force_recreate: bool = False) -> bool:
        """Deploy model to Ollama"""
        logging.info("Starting model deployment to Ollama...")
        
        # Check if Ollama is running
        if not self.ollama_manager.is_ollama_running():
            logging.info("Starting Ollama service...")
            if not self.ollama_manager.start_ollama_service():
                logging.error("Failed to start Ollama service")
                return False
            
            # Wait for service to start
            time.sleep(5)
        
        # Check if model already exists
        if self.ollama_manager.model_exists() and not force_recreate:
            logging.info(f"Model {self.ollama_manager.model_name} already exists")
            user_input = input("Model exists. Recreate? (y/N): ").lower().strip()
            if user_input != 'y':
                logging.info("Deployment cancelled")
                return True
        
        # Create Modelfile
        modelfile_path = self.create_modelfile()
        
        # Create/recreate model
        if self.ollama_manager.model_exists():
            logging.info("Deleting existing model...")
            self.ollama_manager.delete_model()
        
        logging.info("Creating new model...")
        success = self.ollama_manager.create_model(modelfile_path)
        
        if success:
            logging.info("Model deployment completed successfully")
            return True
        else:
            logging.error("Model deployment failed")
            return False
    
    def test_deployment(self) -> Dict[str, any]:
        """Test the deployed model"""
        logging.info("Testing deployed model...")
        
        test_cases = [
            {
                "name": "Basic SELECT",
                "schema": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100), created_at TIMESTAMP);",
                "request": "Find all users created in the last 30 days",
                "expected_keywords": ["SELECT", "users", "created_at", "30"]
            },
            {
                "name": "Aggregation Query",
                "schema": "CREATE TABLE orders (id INT, user_id INT, total DECIMAL(10,2), order_date DATE);",
                "request": "Calculate the total revenue by month for this year",
                "expected_keywords": ["SELECT", "SUM", "total", "GROUP BY", "order_date"]
            },
            {
                "name": "JOIN Query",
                "schema": """CREATE TABLE customers (id INT, name VARCHAR(50), email VARCHAR(100));
CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);""",
                "request": "List all customers with their total order amounts, including customers with no orders",
                "expected_keywords": ["SELECT", "LEFT JOIN", "customers", "orders", "SUM"]
            }
        ]
        
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test_case in test_cases:
            try:
                prompt = f"{test_case['schema']}\n\nRequest: {test_case['request']}"
                response = self.ollama_manager.query_model_safe(prompt, timeout=30)
                
                if response and "{ end }<|end|>" not in response:
                    # Extract SQL from response
                    sql_query = SQLValidator.extract_sql_from_text(response)
                    
                    if sql_query:
                        # Validate SQL syntax
                        is_valid, validation_msg = SQLValidator.validate_sql(sql_query)
                        
                        # Check for expected keywords
                        response_upper = response.upper()
                        keywords_found = sum(
                            1 for keyword in test_case['expected_keywords'] 
                            if keyword.upper() in response_upper
                        )
                        keyword_score = keywords_found / len(test_case['expected_keywords'])
                        
                        test_result = {
                            "test_name": test_case['name'],
                            "status": "PASS" if is_valid and keyword_score > 0.5 else "FAIL",
                            "sql_query": sql_query,
                            "is_valid_sql": is_valid,
                            "validation_message": validation_msg,
                            "keyword_score": keyword_score,
                            "full_response": response
                        }
                        
                        if test_result["status"] == "PASS":
                            results["passed"] += 1
                        else:
                            results["failed"] += 1
                    else:
                        test_result = {
                            "test_name": test_case['name'],
                            "status": "FAIL",
                            "error": "No SQL query found in response",
                            "full_response": response
                        }
                        results["failed"] += 1
                else:
                    test_result = {
                        "test_name": test_case['name'],
                        "status": "FAIL",
                        "error": "No response from model"
                    }
                    results["failed"] += 1
                
                results["test_results"].append(test_result)
                logging.info(f"Test '{test_case['name']}': {test_result['status']}")
                
            except Exception as e:
                test_result = {
                    "test_name": test_case['name'],
                    "status": "ERROR",
                    "error": str(e)
                }
                results["test_results"].append(test_result)
                results["failed"] += 1
                logging.error(f"Test '{test_case['name']}' failed with error: {e}")
        
        # Calculate success rate
        results["success_rate"] = results["passed"] / results["total_tests"] if results["total_tests"] > 0 else 0
        
        return results
    
    def run_interactive_demo(self):
        """Run interactive demo of the deployed model"""
        print("\n" + "="*60)
        print("🤖 Text-to-SQL Interactive Demo")
        print("="*60)
        print("Enter your database schema and natural language queries.")
        print("Type 'quit' to exit, 'help' for examples.\n")
        
        while True:
            try:
                print("\n📋 Database Schema:")
                schema = input("Enter your database schema (or 'help' for examples): ").strip()
                
                if schema.lower() == 'quit':
                    break
                elif schema.lower() == 'help':
                    self._show_examples()
                    continue
                elif not schema:
                    print("❌ Please provide a database schema.")
                    continue
                
                print("\n❓ Natural Language Query:")
                query = input("Enter your query: ").strip()
                
                if not query:
                    print("❌ Please provide a query.")
                    continue
                
                print("\n🔄 Generating SQL...")
                prompt = f"{schema}\n\nRequest: {query}"
                response = self.ollama_manager.query_model_safe(prompt, timeout=30)
                
                if response and "{ end }<|end|>" not in response:
                    print("\n✅ Generated Response:")
                    print("-" * 40)
                    print(response)
                    print("-" * 40)
                    
                    # Validate the SQL
                    sql_query = SQLValidator.extract_sql_from_text(response)
                    if sql_query:
                        is_valid, msg = SQLValidator.validate_sql(sql_query)
                        print(f"\n🔍 SQL Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
                        if not is_valid:
                            print(f"   Error: {msg}")
                    else:
                        print("\n⚠️  No SQL query found in response")
                else:
                    print("\n❌ Failed to get response from model")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_examples(self):
        """Show example schemas and queries"""
        examples = [
            {
                "name": "E-commerce Database",
                "schema": """CREATE TABLE customers (id INT, name VARCHAR(50), email VARCHAR(100));
CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50));
CREATE TABLE orders (id INT, customer_id INT, product_id INT, quantity INT, order_date DATE);""",
                "queries": [
                    "Find the top 5 best-selling products",
                    "Get total revenue by customer",
                    "List products that have never been ordered"
                ]
            },
            {
                "name": "HR Database",
                "schema": """CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(50), salary DECIMAL(10,2), hire_date DATE);
CREATE TABLE departments (id INT, name VARCHAR(50), budget DECIMAL(12,2));""",
                "queries": [
                    "Find employees hired in the last year",
                    "Calculate average salary by department",
                    "List departments over budget"
                ]
            }
        ]
        
        print("\n📚 Example Schemas and Queries:")
        print("="*50)
        
        for example in examples:
            print(f"\n🏢 {example['name']}:")
            print(f"Schema: {example['schema']}")
            print("Sample queries:")
            for query in example['queries']:
                print(f"  • {query}")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Text-to-SQL model to Ollama")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--force", "-f", action="store_true", help="Force recreate model if exists")
    parser.add_argument("--test", "-t", action="store_true", help="Run tests after deployment")
    parser.add_argument("--demo", "-d", action="store_true", help="Run interactive demo after deployment")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deployment, only run tests/demo")
    
    args = parser.parse_args()
    
    # Check system requirements
    reqs = check_system_requirements()
    if not reqs['ollama_available']:
        print("❌ Ollama is not available. Please install Ollama first.")
        print("   Visit: https://ollama.ai/")
        sys.exit(1)
    
    # Initialize deployer
    try:
        deployer = ModelDeployer(args.config)
    except Exception as e:
        logging.error(f"Failed to initialize deployer: {e}")
        sys.exit(1)
    
    success = True
    
    # Deploy model
    if not args.skip_deploy:
        success = deployer.deploy_model(args.force)
        
        if not success:
            logging.error("Deployment failed")
            sys.exit(1)
        
        print("✅ Model deployed successfully!")
    
    # Run tests
    if args.test and success:
        print("\n🧪 Running deployment tests...")
        test_results = deployer.test_deployment()
        
        print(f"\n📊 Test Results:")
        print(f"   Total: {test_results['total_tests']}")
        print(f"   Passed: {test_results['passed']}")
        print(f"   Failed: {test_results['failed']}")
        print(f"   Success Rate: {test_results['success_rate']:.1%}")
        
        # Save detailed results
        import json
        with open("deployment_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print(f"📄 Detailed results saved to deployment_test_results.json")
    
    # Run interactive demo
    if args.demo and success:
        deployer.run_interactive_demo()

if __name__ == "__main__":
    main()
