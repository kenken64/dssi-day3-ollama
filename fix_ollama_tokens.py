#!/usr/bin/env python3
"""
Fix script for Ollama token generation issues
Resolves the { end }<|end|> repetition problem
"""

import subprocess
import sys
import os
from pathlib import Path

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first.")
        return False

def delete_problematic_model(model_name="text-to-sql"):
    """Delete the problematic model"""
    print(f"🗑️  Deleting problematic model: {model_name}")
    try:
        result = subprocess.run(["ollama", "rm", model_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Model {model_name} deleted successfully")
            return True
        else:
            print(f"⚠️  Model {model_name} not found or already deleted")
            return True
    except Exception as e:
        print(f"❌ Error deleting model: {e}")
        return False

def create_fixed_modelfile():
    """Create a new Modelfile with proper token handling"""
    modelfile_content = """FROM codellama:7b

# Improved parameters to prevent token loops
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_predict 256
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "<|end|>"
PARAMETER stop "{ end }"
PARAMETER stop "END"
PARAMETER stop "```"

# Simplified system prompt
SYSTEM \"\"\"You are a SQL query generator. Convert natural language to SQL queries.

Rules:
1. Generate only valid SQL code
2. Use the provided database schema
3. Be concise and accurate
4. Stop after generating the query

Format: Provide the SQL query in a code block, then briefly explain it.\"\"\"

# Simple template without complex tokens
TEMPLATE \"\"\"{{ .System }}

{{ .Prompt }}

SQL Query:\"\"\"
"""
    
    with open("Modelfile.fixed", "w") as f:
        f.write(modelfile_content)
    
    print("✅ Created fixed Modelfile")
    return "Modelfile.fixed"

def recreate_model(model_name="text-to-sql"):
    """Recreate the model with fixed configuration"""
    print(f"🔧 Creating fixed model: {model_name}")
    
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", "Modelfile.fixed"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Model {model_name} created successfully")
            return True
        else:
            print(f"❌ Failed to create model: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def test_fixed_model(model_name="text-to-sql"):
    """Test the fixed model with a simple query"""
    print(f"🧪 Testing fixed model: {model_name}")
    
    test_prompt = """CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100));

Generate a SQL query to find all users with Gmail email addresses."""
    
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=test_prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = result.stdout.strip()
            
            # Check for token loop issue
            if "{ end }<|end|>" in response or response.count("<|end|>") > 3:
                print("❌ Model still has token issues")
                print(f"Response: {response[:200]}...")
                return False
            else:
                print("✅ Model test successful")
                print(f"Response: {response[:200]}...")
                return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main fix routine"""
    print("🔧 Ollama Token Fix Script")
    print("=" * 40)
    
    # Check Ollama service
    if not check_ollama_service():
        sys.exit(1)
    
    model_name = input("Enter model name to fix (default: text-to-sql): ").strip()
    if not model_name:
        model_name = "text-to-sql"
    
    print(f"\n🎯 Fixing model: {model_name}")
    
    # Step 1: Delete problematic model
    if not delete_problematic_model(model_name):
        print("❌ Failed to delete model")
        sys.exit(1)
    
    # Step 2: Create fixed Modelfile
    modelfile_path = create_fixed_modelfile()
    
    # Step 3: Recreate model
    if not recreate_model(model_name):
        print("❌ Failed to recreate model")
        sys.exit(1)
    
    # Step 4: Test fixed model
    if test_fixed_model(model_name):
        print("\n✅ Model fixed successfully!")
        print(f"You can now use: ollama run {model_name}")
    else:
        print("\n❌ Model still has issues. You may need to:")
        print("1. Try a different base model (e.g., llama2:7b)")
        print("2. Check your Ollama version")
        print("3. Restart Ollama service")
    
    # Cleanup
    if os.path.exists("Modelfile.fixed"):
        os.remove("Modelfile.fixed")

if __name__ == "__main__":
    main()
