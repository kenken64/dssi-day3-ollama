#!/usr/bin/env python3
"""
Quick Start Examples for Text-to-SQL Model
Run these examples to quickly test the model functionality
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from utils import OllamaManager, SQLValidator, ConfigManager

def check_setup():
    """Check if the system is properly set up"""
    print("🔍 Checking system setup...")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
        else:
            print("❌ Ollama is not installed. Please install from https://ollama.ai/")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed. Please install from https://ollama.ai/")
        return False
    
    # Check if model exists
    manager = OllamaManager("text-to-sql")
    if manager.model_exists():
        print("✅ text-to-sql model is available")
    else:
        print("❌ text-to-sql model not found. Please run training and deployment first.")
        print("   Use: python text_to_sql_train.py --mode full --max-samples 1000")
        return False
    
    print("✅ System setup is complete!")
    return True

def example_basic_queries():
    """Run basic SQL query examples"""
    print("\n🚀 Running Basic Query Examples")
    print("=" * 50)
    
    manager = OllamaManager("text-to-sql")
    
    examples = [
        {
            "name": "Simple SELECT",
            "schema": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100), created_at DATE);",
            "query": "Show all users"
        },
        {
            "name": "Filtered SELECT",
            "schema": "CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50));",
            "query": "Find all products in the Electronics category"
        },
        {
            "name": "COUNT Aggregation",
            "schema": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);",
            "query": "How many orders were placed?"
        },
        {
            "name": "Date Filtering",
            "schema": "CREATE TABLE sales (id INT, product_id INT, sale_date DATE, amount DECIMAL(10,2));",
            "query": "Find sales from the last 30 days"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example['name']}")
        print("-" * 30)
        print(f"Schema: {example['schema']}")
        print(f"Query: {example['query']}")
        
        prompt = f"{example['schema']}\n\nRequest: {example['query']}"
        
        start_time = time.time()
        response = manager.query_model(prompt, timeout=30)
        end_time = time.time()
        
        if response:
            print(f"Response ({end_time - start_time:.2f}s):")
            print(response)
            
            # Validate SQL
            sql_query = SQLValidator.extract_sql_from_text(response)
            if sql_query:
                is_valid, msg = SQLValidator.validate_sql(sql_query)
                print(f"SQL Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
                if not is_valid:
                    print(f"Error: {msg}")
            else:
                print("⚠️ No SQL found in response")
        else:
            print("❌ No response received")
        
        print()

def example_complex_queries():
    """Run complex SQL query examples"""
    print("\n🎯 Running Complex Query Examples")
    print("=" * 50)
    
    manager = OllamaManager("text-to-sql")
    
    complex_examples = [
        {
            "name": "JOIN Query",
            "schema": """CREATE TABLE customers (id INT, name VARCHAR(50), email VARCHAR(100));
CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);""",
            "query": "Show customer names with their total order amounts"
        },
        {
            "name": "GROUP BY with Aggregation",
            "schema": "CREATE TABLE sales (id INT, product_id INT, quantity INT, price DECIMAL(10,2), sale_date DATE);",
            "query": "Calculate total revenue by month for this year"
        },
        {
            "name": "Subquery",
            "schema": "CREATE TABLE employees (id INT, name VARCHAR(50), salary DECIMAL(10,2), department VARCHAR(50));",
            "query": "Find employees who earn more than the average salary"
        },
        {
            "name": "Multiple JOINs",
            "schema": """CREATE TABLE customers (id INT, name VARCHAR(50));
CREATE TABLE orders (id INT, customer_id INT, order_date DATE);
CREATE TABLE order_items (id INT, order_id INT, product_id INT, quantity INT);
CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2));""",
            "query": "List customers with their total order value including product details"
        }
    ]
    
    for i, example in enumerate(complex_examples, 1):
        print(f"\n🔧 Complex Example {i}: {example['name']}")
        print("-" * 40)
        print(f"Schema: {example['schema']}")
        print(f"Query: {example['query']}")
        
        prompt = f"{example['schema']}\n\nRequest: {example['query']}"
        
        start_time = time.time()
        response = manager.query_model(prompt, timeout=45)
        end_time = time.time()
        
        if response:
            print(f"Response ({end_time - start_time:.2f}s):")
            print(response)
            
            # Validate SQL
            sql_query = SQLValidator.extract_sql_from_text(response)
            if sql_query:
                is_valid, msg = SQLValidator.validate_sql(sql_query)
                print(f"SQL Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
                if not is_valid:
                    print(f"Error: {msg}")
            else:
                print("⚠️ No SQL found in response")
        else:
            print("❌ No response received")
        
        print()

def example_domain_specific():
    """Run domain-specific examples"""
    print("\n🏢 Running Domain-Specific Examples")
    print("=" * 50)
    
    manager = OllamaManager("text-to-sql")
    
    domain_examples = [
        {
            "domain": "E-commerce",
            "schema": """CREATE TABLE customers (id INT, name VARCHAR(50), registration_date DATE, country VARCHAR(50));
CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50));
CREATE TABLE orders (id INT, customer_id INT, product_id INT, quantity INT, order_date DATE);""",
            "query": "Find the top 5 best-selling products by revenue in the last quarter"
        },
        {
            "domain": "Healthcare",
            "schema": """CREATE TABLE patients (id INT, name VARCHAR(50), date_of_birth DATE, gender VARCHAR(10));
CREATE TABLE doctors (id INT, name VARCHAR(50), specialty VARCHAR(50));
CREATE TABLE appointments (id INT, patient_id INT, doctor_id INT, appointment_date DATE, duration INT);""",
            "query": "Show the average appointment duration by doctor specialty"
        },
        {
            "domain": "Finance",
            "schema": """CREATE TABLE accounts (id INT, account_number VARCHAR(20), account_type VARCHAR(20), balance DECIMAL(15,2));
CREATE TABLE transactions (id INT, account_id INT, transaction_date DATE, amount DECIMAL(15,2), transaction_type VARCHAR(20));""",
            "query": "Calculate the total deposits and withdrawals by account type for the current month"
        },
        {
            "domain": "HR Management",
            "schema": """CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(50), position VARCHAR(50), salary DECIMAL(10,2), hire_date DATE);
CREATE TABLE performance_reviews (id INT, employee_id INT, review_date DATE, score INT, reviewer_id INT);""",
            "query": "Find employees with performance scores above 8 who were hired in the last two years"
        }
    ]
    
    for example in domain_examples:
        print(f"\n🏷️ {example['domain']} Domain")
        print("-" * 30)
        print(f"Schema: {example['schema']}")
        print(f"Query: {example['query']}")
        
        prompt = f"Domain: {example['domain']}\n\n{example['schema']}\n\nRequest: {example['query']}"
        
        start_time = time.time()
        response = manager.query_model(prompt, timeout=45)
        end_time = time.time()
        
        if response:
            print(f"Response ({end_time - start_time:.2f}s):")
            print(response)
            
            # Validate SQL
            sql_query = SQLValidator.extract_sql_from_text(response)
            if sql_query:
                is_valid, msg = SQLValidator.validate_sql(sql_query)
                print(f"SQL Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        else:
            print("❌ No response received")
        
        print()

def interactive_mode():
    """Run interactive mode for custom queries"""
    print("\n💬 Interactive Mode")
    print("=" * 50)
    print("Enter your own database schema and queries!")
    print("Type 'quit' to exit, 'examples' to see sample schemas")
    
    manager = OllamaManager("text-to-sql")
    
    while True:
        try:
            print("\n" + "─" * 50)
            schema = input("📋 Enter database schema: ").strip()
            
            if schema.lower() == 'quit':
                break
            elif schema.lower() == 'examples':
                show_schema_examples()
                continue
            elif not schema:
                print("❌ Please provide a schema")
                continue
            
            query = input("❓ Enter your query: ").strip()
            
            if not query:
                print("❌ Please provide a query")
                continue
            
            print("\n🔄 Generating SQL...")
            prompt = f"{schema}\n\nRequest: {query}"
            
            start_time = time.time()
            response = manager.query_model(prompt, timeout=30)
            end_time = time.time()
            
            if response:
                print(f"\n✅ Response ({end_time - start_time:.2f}s):")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                # Validate SQL
                sql_query = SQLValidator.extract_sql_from_text(response)
                if sql_query:
                    is_valid, msg = SQLValidator.validate_sql(sql_query)
                    print(f"\n🔍 SQL Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
                    if not is_valid:
                        print(f"   Error: {msg}")
                else:
                    print("\n⚠️ No SQL query found in response")
            else:
                print("\n❌ Failed to get response")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def show_schema_examples():
    """Show example database schemas"""
    examples = {
        "Simple User Management": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100), created_at DATE);",
        
        "E-commerce": """CREATE TABLE customers (id INT, name VARCHAR(50), email VARCHAR(100));
CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50));
CREATE TABLE orders (id INT, customer_id INT, product_id INT, quantity INT, order_date DATE);""",
        
        "Library System": """CREATE TABLE books (id INT, title VARCHAR(200), author VARCHAR(100), isbn VARCHAR(20), publication_year INT);
CREATE TABLE members (id INT, name VARCHAR(50), email VARCHAR(100), membership_date DATE);
CREATE TABLE loans (id INT, book_id INT, member_id INT, loan_date DATE, return_date DATE);""",
        
        "School Management": """CREATE TABLE students (id INT, name VARCHAR(50), grade INT, enrollment_date DATE);
CREATE TABLE courses (id INT, name VARCHAR(100), credits INT, instructor VARCHAR(50));
CREATE TABLE enrollments (id INT, student_id INT, course_id INT, enrollment_date DATE, grade VARCHAR(2));"""
    }
    
    print("\n📚 Example Database Schemas:")
    print("=" * 40)
    
    for name, schema in examples.items():
        print(f"\n🏷️ {name}:")
        print(schema)

def performance_test():
    """Run performance tests"""
    print("\n⚡ Performance Testing")
    print("=" * 50)
    
    manager = OllamaManager("text-to-sql")
    
    test_cases = [
        ("Simple", "CREATE TABLE users (id INT, name VARCHAR(50));", "Get all users"),
        ("Medium", "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2)); CREATE TABLE customers (id INT, name VARCHAR(50));", "Show customers with their total orders"),
        ("Complex", "CREATE TABLE products (id INT, name VARCHAR(100), category VARCHAR(50)); CREATE TABLE order_items (order_id INT, product_id INT, quantity INT); CREATE TABLE orders (id INT, customer_id INT, order_date DATE);", "Find the top 10 best-selling products by category"),
    ]
    
    results = []
    
    for complexity, schema, query in test_cases:
        print(f"\n🧪 Testing {complexity} Query...")
        prompt = f"{schema}\n\nRequest: {query}"
        
        times = []
        successes = 0
        
        # Run multiple iterations
        for i in range(3):
            start_time = time.time()
            response = manager.query_model(prompt, timeout=60)
            end_time = time.time()
            
            response_time = end_time - start_time
            times.append(response_time)
            
            if response and SQLValidator.extract_sql_from_text(response):
                successes += 1
            
            print(f"  Iteration {i+1}: {response_time:.2f}s")
        
        avg_time = sum(times) / len(times)
        success_rate = successes / len(times)
        
        results.append({
            "complexity": complexity,
            "avg_time": avg_time,
            "success_rate": success_rate,
            "times": times
        })
        
        print(f"  Average: {avg_time:.2f}s, Success Rate: {success_rate:.1%}")
    
    print(f"\n📊 Performance Summary:")
    print("=" * 30)
    for result in results:
        print(f"{result['complexity']:8} | {result['avg_time']:6.2f}s | {result['success_rate']:6.1%}")

def save_results_to_file():
    """Save example results to a file for later reference"""
    print("\n💾 Saving Results")
    print("=" * 50)
    
    # This would run a subset of examples and save results
    manager = OllamaManager("text-to-sql")
    
    test_queries = [
        {
            "schema": "CREATE TABLE employees (id INT, name VARCHAR(50), department VARCHAR(50), salary DECIMAL(10,2));",
            "query": "Find employees in Engineering department"
        },
        {
            "schema": "CREATE TABLE sales (id INT, amount DECIMAL(10,2), sale_date DATE);",
            "query": "Calculate total sales for this month"
        }
    ]
    
    results = []
    
    for test in test_queries:
        prompt = f"{test['schema']}\n\nRequest: {test['query']}"
        response = manager.query_model(prompt)
        
        results.append({
            "schema": test['schema'],
            "query": test['query'],
            "response": response,
            "sql": SQLValidator.extract_sql_from_text(response) if response else None,
            "timestamp": time.time()
        })
    
    # Save to file
    output_file = "example_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_file}")

def main():
    """Main function to run examples"""
    print("🤖 Text-to-SQL Model - Quick Start Examples")
    print("=" * 60)
    
    if not check_setup():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        return
    
    while True:
        print("\n📋 Available Examples:")
        print("1. Basic SQL Queries")
        print("2. Complex SQL Queries") 
        print("3. Domain-Specific Examples")
        print("4. Interactive Mode")
        print("5. Performance Testing")
        print("6. Save Example Results")
        print("0. Exit")
        
        try:
            choice = input("\nSelect an option (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                example_basic_queries()
            elif choice == '2':
                example_complex_queries()
            elif choice == '3':
                example_domain_specific()
            elif choice == '4':
                interactive_mode()
            elif choice == '5':
                performance_test()
            elif choice == '6':
                save_results_to_file()
            else:
                print("❌ Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
