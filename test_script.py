#!/usr/bin/env python3
"""
Comprehensive testing suite for Text-to-SQL model
"""

import os
import sys
import json
import argparse
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import unittest
from dataclasses import dataclass

from utils import (
    ConfigManager,
    OllamaManager,
    ModelEvaluator,
    SQLValidator,
    setup_logging
)

@dataclass
class TestCase:
    """Test case for SQL generation"""
    name: str
    schema: str
    natural_query: str
    expected_sql: Optional[str] = None
    expected_keywords: Optional[List[str]] = None
    complexity: str = "basic"
    domain: str = "general"

class SQLTestDatabase:
    """Create and manage test databases for validation"""
    
    def __init__(self):
        self.connections = {}
        self.setup_test_databases()
    
    def setup_test_databases(self):
        """Setup various test databases"""
        self.setup_ecommerce_db()
        self.setup_hr_db()
        self.setup_finance_db()
        self.setup_healthcare_db()
    
    def setup_ecommerce_db(self):
        """Setup e-commerce test database"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                registration_date DATE,
                country TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price DECIMAL(10,2),
                category TEXT,
                stock_quantity INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date DATE,
                total_amount DECIMAL(10,2),
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                unit_price DECIMAL(10,2),
                FOREIGN KEY (order_id) REFERENCES orders (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)
        
        # Insert sample data
        sample_data = [
            # Customers
            "INSERT INTO customers VALUES (1, 'John Doe', 'john@email.com', '2023-01-15', 'USA')",
            "INSERT INTO customers VALUES (2, 'Jane Smith', 'jane@email.com', '2023-02-20', 'UK')",
            "INSERT INTO customers VALUES (3, 'Bob Johnson', 'bob@email.com', '2023-03-10', 'Canada')",
            
            # Products
            "INSERT INTO products VALUES (1, 'Laptop', 999.99, 'Electronics', 50)",
            "INSERT INTO products VALUES (2, 'Mouse', 29.99, 'Electronics', 200)",
            "INSERT INTO products VALUES (3, 'Book', 19.99, 'Books', 100)",
            
            # Orders
            "INSERT INTO orders VALUES (1, 1, '2023-04-01', 1029.98, 'completed')",
            "INSERT INTO orders VALUES (2, 2, '2023-04-02', 19.99, 'completed')",
            "INSERT INTO orders VALUES (3, 1, '2023-04-03', 29.99, 'pending')",
            
            # Order Items
            "INSERT INTO order_items VALUES (1, 1, 1, 1, 999.99)",
            "INSERT INTO order_items VALUES (2, 1, 2, 1, 29.99)",
            "INSERT INTO order_items VALUES (3, 2, 3, 1, 19.99)",
            "INSERT INTO order_items VALUES (4, 3, 2, 1, 29.99)",
        ]
        
        for query in sample_data:
            cursor.execute(query)
        
        conn.commit()
        self.connections['ecommerce'] = conn
    
    def setup_hr_db(self):
        """Setup HR test database"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                salary DECIMAL(10,2),
                hire_date DATE,
                manager_id INTEGER,
                FOREIGN KEY (manager_id) REFERENCES employees (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget DECIMAL(12,2),
                location TEXT
            )
        """)
        
        # Insert sample data
        sample_data = [
            "INSERT INTO departments VALUES (1, 'Engineering', 2000000, 'San Francisco')",
            "INSERT INTO departments VALUES (2, 'Marketing', 800000, 'New York')",
            "INSERT INTO departments VALUES (3, 'Sales', 1200000, 'Chicago')",
            
            "INSERT INTO employees VALUES (1, 'Alice Johnson', 'Engineering', 'Senior Developer', 95000, '2020-01-15', NULL)",
            "INSERT INTO employees VALUES (2, 'Bob Smith', 'Engineering', 'Junior Developer', 65000, '2022-03-01', 1)",
            "INSERT INTO employees VALUES (3, 'Carol Brown', 'Marketing', 'Marketing Manager', 75000, '2021-06-15', NULL)",
            "INSERT INTO employees VALUES (4, 'David Wilson', 'Sales', 'Sales Rep', 55000, '2023-01-01', NULL)",
        ]
        
        for query in sample_data:
            cursor.execute(query)
        
        conn.commit()
        self.connections['hr'] = conn
    
    def setup_finance_db(self):
        """Setup finance test database"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY,
                account_number TEXT UNIQUE,
                account_type TEXT,
                balance DECIMAL(15,2),
                customer_id INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY,
                account_id INTEGER,
                transaction_date DATE,
                amount DECIMAL(15,2),
                transaction_type TEXT,
                description TEXT,
                FOREIGN KEY (account_id) REFERENCES accounts (id)
            )
        """)
        
        # Insert sample data
        sample_data = [
            "INSERT INTO accounts VALUES (1, 'ACC001', 'checking', 5000.00, 101)",
            "INSERT INTO accounts VALUES (2, 'ACC002', 'savings', 15000.00, 101)",
            "INSERT INTO accounts VALUES (3, 'ACC003', 'checking', 3200.00, 102)",
            
            "INSERT INTO transactions VALUES (1, 1, '2023-04-01', -50.00, 'debit', 'ATM Withdrawal')",
            "INSERT INTO transactions VALUES (2, 1, '2023-04-02', 1000.00, 'credit', 'Salary Deposit')",
            "INSERT INTO transactions VALUES (3, 2, '2023-04-01', -200.00, 'debit', 'Transfer to Checking')",
            "INSERT INTO transactions VALUES (4, 3, '2023-04-03', -25.00, 'debit', 'Coffee Shop')",
        ]
        
        for query in sample_data:
            cursor.execute(query)
        
        conn.commit()
        self.connections['finance'] = conn
    
    def setup_healthcare_db(self):
        """Setup healthcare test database"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                date_of_birth DATE,
                gender TEXT,
                phone TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE doctors (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                specialty TEXT,
                department TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE appointments (
                id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                doctor_id INTEGER,
                appointment_date DATE,
                duration_minutes INTEGER,
                status TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id),
                FOREIGN KEY (doctor_id) REFERENCES doctors (id)
            )
        """)
        
        # Insert sample data
        sample_data = [
            "INSERT INTO patients VALUES (1, 'John Patient', '1985-05-15', 'M', '555-0101')",
            "INSERT INTO patients VALUES (2, 'Jane Patient', '1990-08-22', 'F', '555-0102')",
            
            "INSERT INTO doctors VALUES (1, 'Dr. Smith', 'Cardiology', 'Internal Medicine')",
            "INSERT INTO doctors VALUES (2, 'Dr. Johnson', 'Orthopedics', 'Surgery')",
            
            "INSERT INTO appointments VALUES (1, 1, 1, '2023-04-15', 30, 'completed')",
            "INSERT INTO appointments VALUES (2, 2, 2, '2023-04-16', 45, 'scheduled')",
            "INSERT INTO appointments VALUES (3, 1, 2, '2023-04-20', 30, 'scheduled')",
        ]
        
        for query in sample_data:
            cursor.execute(query)
        
        conn.commit()
        self.connections['healthcare'] = conn
    
    def get_connection(self, db_name: str) -> sqlite3.Connection:
        """Get database connection"""
        return self.connections.get(db_name)
    
    def validate_query(self, sql_query: str, db_name: str) -> Tuple[bool, str, Any]:
        """Validate and execute query against test database"""
        conn = self.get_connection(db_name)
        if not conn:
            return False, f"Database {db_name} not found", None
        
        try:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return True, "Query executed successfully", results
        except Exception as e:
            return False, f"Query execution failed: {str(e)}", None

class TextToSQLTester:
    """Main testing class for text-to-SQL functionality"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.ollama_manager = OllamaManager(
            self.config.get('ollama.model_name', 'text-to-sql')
        )
        self.test_db = SQLTestDatabase()
        self.evaluator = ModelEvaluator()
        
        setup_logging(
            self.config.get('logging.level', 'INFO'),
            'testing.log'
        )
    
    def create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases"""
        test_cases = [
            # Basic SELECT queries
            TestCase(
                name="basic_select_all",
                schema="CREATE TABLE customers (id INT, name VARCHAR(50), email VARCHAR(100));",
                natural_query="Show all customers",
                expected_sql="SELECT * FROM customers;",
                expected_keywords=["SELECT", "*", "FROM", "customers"],
                complexity="basic"
            ),
            
            TestCase(
                name="basic_select_with_where",
                schema="CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50));",
                natural_query="Find all products in the Electronics category",
                expected_sql="SELECT * FROM products WHERE category = 'Electronics';",
                expected_keywords=["SELECT", "FROM", "products", "WHERE", "category", "Electronics"],
                complexity="basic"
            ),
            
            # Aggregation queries
            TestCase(
                name="count_aggregation",
                schema="CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);",
                natural_query="How many orders were placed?",
                expected_sql="SELECT COUNT(*) FROM orders;",
                expected_keywords=["SELECT", "COUNT", "FROM", "orders"],
                complexity="aggregation"
            ),
            
            TestCase(
                name="sum_aggregation",
                schema="CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);",
                natural_query="What is the total revenue from all orders?",
                expected_sql="SELECT SUM(total) FROM orders;",
                expected_keywords=["SELECT", "SUM", "total", "FROM", "orders"],
                complexity="aggregation"
            ),
            
            TestCase(
                name="group_by_aggregation",
                schema="CREATE TABLE sales (id INT, product_id INT, quantity INT, sale_date DATE);",
                natural_query="Show total quantity sold by product",
                expected_sql="SELECT product_id, SUM(quantity) FROM sales GROUP BY product_id;",
                expected_keywords=["SELECT", "product_id", "SUM", "quantity", "GROUP BY"],
                complexity="aggregation"
            ),
            
            # JOIN queries
            TestCase(
                name="inner_join",
                schema="""CREATE TABLE customers (id INT, name VARCHAR(50));
                         CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2));""",
                natural_query="Show customer names with their order totals",
                expected_sql="SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id;",
                expected_keywords=["SELECT", "JOIN", "customers", "orders", "ON"],
                complexity="single join"
            ),
            
            TestCase(
                name="left_join",
                schema="""CREATE TABLE customers (id INT, name VARCHAR(50));
                         CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2));""",
                natural_query="List all customers and their order totals, including customers with no orders",
                expected_sql="SELECT c.name, o.total FROM customers c LEFT JOIN orders o ON c.id = o.customer_id;",
                expected_keywords=["SELECT", "LEFT JOIN", "customers", "orders"],
                complexity="single join"
            ),
            
            # Complex queries
            TestCase(
                name="subquery",
                schema="""CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(50), salary DECIMAL(10,2));""",
                natural_query="Find employees who earn more than the average salary",
                expected_sql="SELECT * FROM employees WHERE salary > (SELECT AVG(salary) FROM employees);",
                expected_keywords=["SELECT", "WHERE", "salary", "AVG", "subquery"],
                complexity="subqueries"
            ),
            
            TestCase(
                name="date_filtering",
                schema="CREATE TABLE orders (id INT, customer_id INT, order_date DATE, total DECIMAL(10,2));",
                natural_query="Find orders placed in the last 30 days",
                expected_sql="SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);",
                expected_keywords=["SELECT", "WHERE", "order_date", "DATE", "30"],
                complexity="basic"
            ),
            
            # Domain-specific queries
            TestCase(
                name="ecommerce_revenue_by_month",
                schema="""CREATE TABLE orders (id INT, order_date DATE, total DECIMAL(10,2));""",
                natural_query="Calculate total revenue by month for this year",
                expected_sql="SELECT DATE_FORMAT(order_date, '%Y-%m') as month, SUM(total) FROM orders WHERE YEAR(order_date) = YEAR(CURDATE()) GROUP BY month;",
                expected_keywords=["SELECT", "DATE_FORMAT", "SUM", "GROUP BY", "YEAR"],
                complexity="aggregation",
                domain="ecommerce"
            ),
            
            TestCase(
                name="hr_department_headcount",
                schema="CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(50), hire_date DATE);",
                natural_query="Show headcount by department",
                expected_sql="SELECT department, COUNT(*) as headcount FROM employees GROUP BY department;",
                expected_keywords=["SELECT", "department", "COUNT", "GROUP BY"],
                complexity="aggregation",
                domain="hr"
            ),
        ]
        
        return test_cases
    
    def run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case"""
        logging.info(f"Running test case: {test_case.name}")
        
        result = {
            "test_name": test_case.name,
            "complexity": test_case.complexity,
            "domain": test_case.domain,
            "status": "UNKNOWN",
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Generate SQL using the model
            prompt = f"{test_case.schema}\n\nRequest: {test_case.natural_query}"
            response = self.ollama_manager.query_model(prompt, timeout=30)
            
            if not response:
                result["status"] = "FAIL"
                result["errors"].append("No response from model")
                return result
            
            result["full_response"] = response
            
            # Extract SQL from response
            generated_sql = SQLValidator.extract_sql_from_text(response)
            
            if not generated_sql:
                result["status"] = "FAIL"
                result["errors"].append("No SQL query found in response")
                return result
            
            result["generated_sql"] = generated_sql
            
            # Validate SQL syntax
            is_valid, validation_msg = SQLValidator.validate_sql(generated_sql)
            result["metrics"]["is_syntactically_valid"] = is_valid
            
            if not is_valid:
                result["errors"].append(f"Invalid SQL syntax: {validation_msg}")
            
            # Check for expected keywords
            if test_case.expected_keywords:
                response_upper = response.upper()
                keywords_found = [
                    keyword for keyword in test_case.expected_keywords
                    if keyword.upper() in response_upper
                ]
                keyword_score = len(keywords_found) / len(test_case.expected_keywords)
                result["metrics"]["keyword_match_score"] = keyword_score
                result["keywords_found"] = keywords_found
                result["keywords_missing"] = [
                    kw for kw in test_case.expected_keywords if kw not in keywords_found
                ]
            
            # Test query execution if possible
            db_name = test_case.domain if test_case.domain != "general" else "ecommerce"
            can_execute, exec_msg, exec_results = self.test_db.validate_query(generated_sql, db_name)
            
            result["metrics"]["is_executable"] = can_execute
            if can_execute:
                result["execution_results"] = str(exec_results)
            else:
                result["errors"].append(f"Query execution failed: {exec_msg}")
            
            # Determine overall status
            if is_valid and can_execute:
                if test_case.expected_keywords:
                    result["status"] = "PASS" if keyword_score >= 0.7 else "PARTIAL"
                else:
                    result["status"] = "PASS"
            else:
                result["status"] = "FAIL"
            
        except Exception as e:
            result["status"] = "ERROR"
            result["errors"].append(f"Test execution error: {str(e)}")
            logging.error(f"Error in test case {test_case.name}: {e}")
        
        return result
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Run performance tests"""
        logging.info("Running performance tests...")
        
        performance_cases = [
            ("Simple Query", "CREATE TABLE users (id INT, name VARCHAR(50));", "Get all users"),
            ("Medium Query", """CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);
                              CREATE TABLE customers (id INT, name VARCHAR(50));""", 
             "Show customers with their total order amounts"),
            ("Complex Query", """CREATE TABLE products (id INT, name VARCHAR(100), category VARCHAR(50), price DECIMAL(10,2));
                               CREATE TABLE order_items (order_id INT, product_id INT, quantity INT);
                               CREATE TABLE orders (id INT, customer_id INT, order_date DATE);""",
             "Find the top 5 best-selling products by revenue in the last year"),
        ]
        
        results = {
            "total_tests": len(performance_cases),
            "response_times": [],
            "average_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0,
            "test_details": []
        }
        
        for name, schema, query in performance_cases:
            try:
                import time
                start_time = time.time()
                
                prompt = f"{schema}\n\nRequest: {query}"
                response = self.ollama_manager.query_model(prompt, timeout=60)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                results["response_times"].append(response_time)
                results["min_response_time"] = min(results["min_response_time"], response_time)
                results["max_response_time"] = max(results["max_response_time"], response_time)
                
                results["test_details"].append({
                    "test_name": name,
                    "response_time": response_time,
                    "response_received": response is not None,
                    "response_length": len(response) if response else 0
                })
                
            except Exception as e:
                logging.error(f"Performance test {name} failed: {e}")
                results["test_details"].append({
                    "test_name": name,
                    "error": str(e)
                })
        
        if results["response_times"]:
            results["average_response_time"] = sum(results["response_times"]) / len(results["response_times"])
        
        return results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logging.info("Starting comprehensive test suite...")
        
        # Check if model is available
        if not self.ollama_manager.model_exists():
            return {
                "error": f"Model {self.ollama_manager.model_name} not found in Ollama",
                "suggestion": "Please deploy the model first using deploy.py"
            }
        
        test_cases = self.create_test_cases()
        
        suite_results = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.ollama_manager.model_name,
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "partial": 0,
            "errors": 0,
            "test_results": [],
            "summary_by_complexity": {},
            "summary_by_domain": {},
            "performance_results": {},
            "overall_metrics": {}
        }
        
        # Run individual test cases
        for test_case in test_cases:
            result = self.run_test_case(test_case)
            suite_results["test_results"].append(result)
            
            # Update counters
            status = result["status"]
            if status == "PASS":
                suite_results["passed"] += 1
            elif status == "FAIL":
                suite_results["failed"] += 1
            elif status == "PARTIAL":
                suite_results["partial"] += 1
            else:  # ERROR
                suite_results["errors"] += 1
            
            # Update complexity summary
            complexity = result["complexity"]
            if complexity not in suite_results["summary_by_complexity"]:
                suite_results["summary_by_complexity"][complexity] = {"total": 0, "passed": 0}
            
            suite_results["summary_by_complexity"][complexity]["total"] += 1
            if status == "PASS":
                suite_results["summary_by_complexity"][complexity]["passed"] += 1
            
            # Update domain summary
            domain = result["domain"]
            if domain not in suite_results["summary_by_domain"]:
                suite_results["summary_by_domain"][domain] = {"total": 0, "passed": 0}
            
            suite_results["summary_by_domain"][domain]["total"] += 1
            if status == "PASS":
                suite_results["summary_by_domain"][domain]["passed"] += 1
        
        # Run performance tests
        suite_results["performance_results"] = self.run_performance_test()
        
        # Calculate overall metrics
        total_tests = suite_results["total_tests"]
        if total_tests > 0:
            suite_results["overall_metrics"] = {
                "success_rate": suite_results["passed"] / total_tests,
                "failure_rate": suite_results["failed"] / total_tests,
                "error_rate": suite_results["errors"] / total_tests,
                "partial_success_rate": suite_results["partial"] / total_tests
            }
        
        return suite_results
    
    def generate_test_report(self, results: Dict[str, Any], output_file: str = "test_report.html"):
        """Generate a comprehensive HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Text-to-SQL Model Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .pass {{ border-left-color: #4CAF50; }}
        .fail {{ border-left-color: #f44336; }}
        .partial {{ border-left-color: #ff9800; }}
        .error {{ border-left-color: #9c27b0; }}
        .sql-code {{ background-color: #f5f5f5; padding: 10px; font-family: monospace; white-space: pre-wrap; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Text-to-SQL Model Test Report</h1>
        <p><strong>Model:</strong> {results.get('model_name', 'Unknown')}</p>
        <p><strong>Timestamp:</strong> {results.get('timestamp', 'Unknown')}</p>
    </div>
    
    <div class="summary">
        <div class="metric-card">
            <h3>Total Tests</h3>
            <p style="font-size: 24px; margin: 0;">{results.get('total_tests', 0)}</p>
        </div>
        <div class="metric-card">
            <h3>Passed</h3>
            <p style="font-size: 24px; margin: 0; color: #4CAF50;">{results.get('passed', 0)}</p>
        </div>
        <div class="metric-card">
            <h3>Failed</h3>
            <p style="font-size: 24px; margin: 0; color: #f44336;">{results.get('failed', 0)}</p>
        </div>
        <div class="metric-card">
            <h3>Success Rate</h3>
            <p style="font-size: 24px; margin: 0;">{results.get('overall_metrics', {}).get('success_rate', 0):.1%}</p>
        </div>
    </div>
    
    <h2>Test Results by Category</h2>
    <table>
        <tr><th>Complexity</th><th>Total</th><th>Passed</th><th>Success Rate</th></tr>
"""
        
        for complexity, stats in results.get('summary_by_complexity', {}).items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            html_content += f"""
        <tr>
            <td>{complexity}</td>
            <td>{stats['total']}</td>
            <td>{stats['passed']}</td>
            <td>{success_rate:.1%}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Individual Test Results</h2>
"""
        
        for test_result in results.get('test_results', []):
            status_class = test_result['status'].lower()
            html_content += f"""
    <div class="test-result {status_class}">
        <h3>{test_result['test_name']} - {test_result['status']}</h3>
        <p><strong>Complexity:</strong> {test_result['complexity']}</p>
        <p><strong>Domain:</strong> {test_result['domain']}</p>
"""
            
            if 'generated_sql' in test_result:
                html_content += f"""
        <p><strong>Generated SQL:</strong></p>
        <div class="sql-code">{test_result['generated_sql']}</div>
"""
            
            if test_result.get('errors'):
                html_content += f"""
        <p><strong>Errors:</strong></p>
        <ul>
"""
                for error in test_result['errors']:
                    html_content += f"<li>{error}</li>"
                html_content += "</ul>"
            
            html_content += "</div>"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Test report generated: {output_file}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Text-to-SQL model")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--output", "-o", default="test_results.json", help="Output file for results")
    parser.add_argument("--report", "-r", default="test_report.html", help="HTML report file")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick test subset")
    
    args = parser.parse_args()
    
    # Initialize tester
    try:
        tester = TextToSQLTester(args.config)
    except Exception as e:
        logging.error(f"Failed to initialize tester: {e}")
        sys.exit(1)
    
    # Run tests
    if args.performance_only:
        print("🚀 Running performance tests...")
        results = {"performance_results": tester.run_performance_test()}
    else:
        print("🧪 Running comprehensive test suite...")
        results = tester.run_comprehensive_test_suite()
    
    # Check for errors
    if "error" in results:
        print(f"❌ {results['error']}")
        if "suggestion" in results:
            print(f"💡 {results['suggestion']}")
        sys.exit(1)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    if not args.performance_only:
        tester.generate_test_report(results, args.report)
    
    # Print summary
    if not args.performance_only:
        print(f"\n📊 Test Summary:")
        print(f"   Total: {results['total_tests']}")
        print(f"   Passed: {results['passed']} ({results.get('overall_metrics', {}).get('success_rate', 0):.1%})")
        print(f"   Failed: {results['failed']}")
        print(f"   Errors: {results['errors']}")
        
        if results.get('performance_results'):
            perf = results['performance_results']
            print(f"\n⚡ Performance:")
            print(f"   Avg Response Time: {perf.get('average_response_time', 0):.2f}s")
            print(f"   Min Response Time: {perf.get('min_response_time', 0):.2f}s")
            print(f"   Max Response Time: {perf.get('max_response_time', 0):.2f}s")
    else:
        perf = results['performance_results']
        print(f"\n⚡ Performance Results:")
        print(f"   Average Response Time: {perf.get('average_response_time', 0):.2f}s")
        print(f"   Min Response Time: {perf.get('min_response_time', 0):.2f}s")
        print(f"   Max Response Time: {perf.get('max_response_time', 0):.2f}s")
    
    print(f"\n📁 Results saved to: {args.output}")
    if not args.performance_only:
        print(f"📊 Report generated: {args.report}")

if __name__ == "__main__":
    main()
