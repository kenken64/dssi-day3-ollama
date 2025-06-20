#!/usr/bin/env python3
"""
Demo script for the Text-to-SQL Web API
Shows how to interact with the Flask web application programmatically
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
API_URL = f"{BASE_URL}/api"

def check_web_app_status():
    """Check if the web application is running"""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("🌐 Web App Status:")
            print(f"  - Ollama Running: {'✅' if data['ollama_running'] else '❌'}")
            print(f"  - Model Available: {'✅' if data['model_exists'] else '❌'}")
            print(f"  - Model Name: {data.get('model_name', 'N/A')}")
            return data['ollama_running'] and data['model_exists']
        else:
            print("❌ Web app returned error")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Web app not accessible: {e}")
        return False

def generate_sql_via_api(schema, query):
    """Generate SQL using the web API"""
    try:
        payload = {
            "schema": schema,
            "query": query
        }
        
        response = requests.post(
            f"{API_URL}/generate-sql",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json() if response.content else {"error": "Unknown error"}
            return {"success": False, "error": error_data.get("error", "API error")}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}

def validate_sql_via_api(sql):
    """Validate SQL using the web API"""
    try:
        payload = {"sql": sql}
        response = requests.post(f"{API_URL}/validate-sql", json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": "Validation failed"}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Validation request failed: {e}"}

def demo_examples():
    """Run demo with example queries"""
    examples = [
        {
            "name": "Simple User Query",
            "schema": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100), created_at TIMESTAMP);",
            "query": "Find all users who registered in the last week"
        },
        {
            "name": "E-commerce Analysis", 
            "schema": """CREATE TABLE customers (id INT, name VARCHAR(100), city VARCHAR(50));
CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL(10,2), order_date DATE);""",
            "query": "Show the top 3 customers by total order value"
        },
        {
            "name": "Employee Report",
            "schema": "CREATE TABLE employees (id INT, name VARCHAR(100), department VARCHAR(50), salary DECIMAL(10,2));",
            "query": "Calculate the average salary by department"
        }
    ]
    
    print("📊 Running Text-to-SQL API Demo")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("-" * 30)
        print(f"Schema: {example['schema'][:60]}...")
        print(f"Query: {example['query']}")
        print("🔄 Generating SQL...")
        
        result = generate_sql_via_api(example['schema'], example['query'])
        
        if result['success']:
            sql_query = result.get('sql_query')
            if sql_query:
                print("✅ Generated SQL:")
                print(f"```sql\n{sql_query}\n```")
                
                # Validate the generated SQL
                validation = validate_sql_via_api(sql_query)
                if validation.get('success'):
                    status = "✅ Valid" if validation.get('is_valid') else "❌ Invalid"
                    print(f"Validation: {status}")
                    if validation.get('message'):
                        print(f"Message: {validation['message']}")
                else:
                    print("⚠️  Could not validate SQL")
            else:
                print("⚠️  No SQL generated")
                print(f"Response: {result.get('full_response', 'No response')[:100]}...")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        if i < len(examples):
            time.sleep(1)  # Brief pause between examples

def main():
    print("🤖 Text-to-SQL Web API Demo")
    print("=" * 40)
    
    # Check if web app is running
    if not check_web_app_status():
        print("\n💡 To start the web app:")
        print("   python start_webapp.py")
        print("\n📖 Or visit the web interface:")
        print(f"   {BASE_URL}")
        return
    
    print("\n✅ Web app is ready!")
    
    # Run demo examples
    demo_examples()
    
    print(f"\n🌐 You can also visit the web interface at: {BASE_URL}")
    print("📚 Check out examples at: {}/examples".format(BASE_URL))

if __name__ == "__main__":
    main()
