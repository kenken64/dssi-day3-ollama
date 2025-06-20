#!/usr/bin/env python3
"""
Startup script for the Text-to-SQL Web Application
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if Flask is installed"""
    try:
        import flask
        return True
    except ImportError:
        return False

def check_ollama():
    """Check if Ollama is running and model exists"""
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "Ollama service not running"
        
        # Check if text-to-sql model exists
        if 'text-to-sql' not in result.stdout:
            return False, "text-to-sql model not found"
        
        return True, "Ready"
    except FileNotFoundError:
        return False, "Ollama not installed"

def install_requirements():
    """Install missing requirements"""
    print("📦 Installing Flask...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask>=2.3.0'], check=True)
        print("✅ Flask installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Flask")
        return False

def main():
    print("🚀 Text-to-SQL Web App Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check Flask installation
    if not check_requirements():
        print("⚠️  Flask not found. Installing...")
        if not install_requirements():
            sys.exit(1)
    else:
        print("✅ Flask is installed")
    
    # Check Ollama and model
    ollama_ok, ollama_msg = check_ollama()
    if ollama_ok:
        print(f"✅ Ollama: {ollama_msg}")
    else:
        print(f"⚠️  Ollama: {ollama_msg}")
        print("\nTo fix this:")
        if "not installed" in ollama_msg:
            print("1. Install Ollama from https://ollama.ai/")
        elif "not running" in ollama_msg:
            print("1. Start Ollama service: ollama serve")
        elif "model not found" in ollama_msg:
            print("1. Deploy the model: python deployment_script.py")
        print("2. Then restart this web app")
        print("\nContinuing anyway (web app will show status warnings)...")
    
    # Start the web application
    print("\n🌐 Starting web application...")
    print("📍 Access the app at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Set Flask environment variables
        os.environ['FLASK_APP'] = 'app.py'
        os.environ['FLASK_ENV'] = 'development'
        
        # Start Flask app
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Shutting down web application...")
    except Exception as e:
        print(f"❌ Error starting web app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
