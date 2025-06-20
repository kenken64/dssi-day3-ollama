#!/usr/bin/env python3
"""
Example script showing different ways to run the Text-to-SQL Web App
with various port configurations
"""

import subprocess
import sys
import time
import os

def run_example(name, command, description):
    """Run an example command"""
    print(f"\n📌 {name}")
    print("-" * 40)
    print(f"Description: {description}")
    print(f"Command: {' '.join(command)}")
    print("\nPress Enter to run this example (or Ctrl+C to skip)...")
    
    try:
        input()
        print("🚀 Starting...")
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\n⏭️  Skipping...")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🌐 Text-to-SQL Web App - Port Configuration Examples")
    print("=" * 60)
    print("This script demonstrates different ways to run the web app")
    print("with various port and configuration options.")
    
    examples = [
        {
            "name": "Default Configuration",
            "command": [sys.executable, "start_webapp.py"],
            "description": "Runs on default port 5000 with host 0.0.0.0"
        },
        {
            "name": "Custom Port 8080",
            "command": [sys.executable, "start_webapp.py", "--port", "8080"],
            "description": "Runs on port 8080, accessible at http://localhost:8080"
        },
        {
            "name": "Custom Host and Port",
            "command": [sys.executable, "start_webapp.py", "--host", "127.0.0.1", "--port", "3000"],
            "description": "Runs on 127.0.0.1:3000, only accessible locally"
        },
        {
            "name": "Debug Mode with Custom Port",
            "command": [sys.executable, "start_webapp.py", "--port", "8080", "--debug"],
            "description": "Runs on port 8080 with debug mode enabled"
        },
        {
            "name": "Custom Model and Port",
            "command": [sys.executable, "start_webapp.py", "--port", "9000", "--model-name", "custom-sql-model"],
            "description": "Runs on port 9000 using a custom Ollama model"
        },
        {
            "name": "Direct App Run",
            "command": [sys.executable, "app.py", "--port", "7000", "--debug"],
            "description": "Runs app.py directly on port 7000 with debug mode"
        }
    ]
    
    print("\n📋 Available Examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
    
    for example in examples:
        run_example(example["name"], example["command"], example["description"])
    
    print("\n✅ Examples completed!")
    print("\n💡 Environment Variable Examples:")
    print("export FLASK_PORT=8080 && python start_webapp.py")
    print("export FLASK_HOST=127.0.0.1 && python start_webapp.py")
    print("export OLLAMA_MODEL_NAME=my-model && python start_webapp.py")
    
    print("\n📊 Testing Examples:")
    print("python demo_webapp.py --port 8080")
    print("curl http://localhost:8080/api/status")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
