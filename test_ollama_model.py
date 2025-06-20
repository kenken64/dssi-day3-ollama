#!/usr/bin/env python3
"""
Quick test script for Ollama text-to-sql model
"""

import subprocess
import sys

def test_ollama_model(model_name="text-to-sql"):
    """Test the Ollama model with a simple query"""
    
    test_cases = [
        {
            "name": "Simple SELECT",
            "prompt": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100));\n\nFind all users with Gmail addresses"
        },
        {
            "name": "COUNT query", 
            "prompt": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2));\n\nCount the total number of orders"
        },
        {
            "name": "JOIN query",
            "prompt": "CREATE TABLE customers (id INT, name VARCHAR(50));\nCREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2));\n\nFind all customers with their order totals"
        }
    ]
    
    print(f"🧪 Testing Ollama model: {model_name}")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = subprocess.run(
                ["ollama", "run", model_name],
                input=test_case['prompt'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Check for token loop issue
                if "{ end }<|end|>" in response:
                    print("❌ FAILED: Token generation loop detected")
                    print(f"Response: {response[:100]}...")
                    return False
                elif len(response) > 1000:
                    print("❌ FAILED: Response too long (possible loop)")
                    print(f"Response length: {len(response)} characters")
                    return False
                elif not response:
                    print("❌ FAILED: Empty response")
                    return False
                else:
                    print("✅ PASSED")
                    print(f"Response: {response[:200]}...")
                    if len(response) > 200:
                        print("... (truncated)")
            else:
                print("❌ FAILED: Model error")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ FAILED: Timeout (possible infinite loop)")
            return False
        except Exception as e:
            print(f"❌ FAILED: Exception - {e}")
            return False
    
    print("\n✅ All tests passed! Model is working correctly.")
    return True

def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "text-to-sql"
    
    # Check if ollama is available
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama is not available. Please install and start Ollama first.")
        sys.exit(1)
    
    # Check if model exists
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_name not in result.stdout:
        print(f"❌ Model '{model_name}' not found.")
        print("Available models:")
        print(result.stdout)
        sys.exit(1)
    
    # Run tests
    if test_ollama_model(model_name):
        print(f"\n🎉 Model '{model_name}' is ready for use!")
    else:
        print(f"\n💡 Model '{model_name}' has issues. Try running: python fix_ollama_tokens.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
