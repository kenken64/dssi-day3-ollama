#!/usr/bin/env python3
"""
Flask Web App for Text-to-SQL Model
Simple web interface to interact with the Ollama text-to-SQL model
"""

from flask import Flask, render_template, request, jsonify, flash
import subprocess
import json
import logging
import os
from datetime import datetime
from utils import OllamaManager, SQLValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change in production

# Initialize Ollama manager
try:
    ollama_manager = OllamaManager("text-to-sql")
except Exception as e:
    logger.error(f"Failed to initialize Ollama manager: {e}")
    ollama_manager = None

@app.route('/')
def index():
    """Main page with the text-to-SQL interface"""
    return render_template('index.html')

@app.route('/api/generate-sql', methods=['POST'])
def generate_sql():
    """API endpoint to generate SQL from natural language"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        schema = data.get('schema', '').strip()
        query = data.get('query', '').strip()
        
        if not schema:
            return jsonify({
                'success': False,
                'error': 'Database schema is required'
            }), 400
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Natural language query is required'
            }), 400
        
        # Check if Ollama manager is available
        if not ollama_manager:
            return jsonify({
                'success': False,
                'error': 'Ollama model not available. Please check if the model is deployed.'
            }), 500
        
        # Format prompt
        prompt = f"Database Schema:\n{schema}\n\nRequest: {query}"
        
        # Generate SQL using the safer method
        logger.info(f"Generating SQL for query: {query[:50]}...")
        
        if hasattr(ollama_manager, 'query_model_safe'):
            response = ollama_manager.query_model_safe(prompt, timeout=30)
        else:
            response = ollama_manager.query_model(prompt, timeout=30)
        
        if not response:
            return jsonify({
                'success': False,
                'error': 'Failed to get response from model'
            }), 500
        
        # Check for token loop issue
        if "{ end }<|end|>" in response:
            return jsonify({
                'success': False,
                'error': 'Model encountered a token generation issue. Please try again or check model configuration.'
            }), 500
        
        # Extract SQL from response
        sql_query = SQLValidator.extract_sql_from_text(response)
        
        if sql_query:
            # Validate SQL syntax
            is_valid, validation_msg = SQLValidator.validate_sql(sql_query)
            
            return jsonify({
                'success': True,
                'sql_query': sql_query,
                'full_response': response,
                'is_valid': is_valid,
                'validation_message': validation_msg,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': True,
                'sql_query': None,
                'full_response': response,
                'is_valid': False,
                'validation_message': 'No SQL query found in response',
                'timestamp': datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

@app.route('/api/validate-sql', methods=['POST'])
def validate_sql():
    """API endpoint to validate SQL syntax"""
    try:
        data = request.get_json()
        sql_query = data.get('sql', '').strip()
        
        if not sql_query:
            return jsonify({
                'success': False,
                'error': 'SQL query is required'
            })
        
        is_valid, message = SQLValidator.validate_sql(sql_query)
        
        return jsonify({
            'success': True,
            'is_valid': is_valid,
            'message': message
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/status')
def status():
    """Check the status of the text-to-SQL service"""
    try:
        # Check if Ollama is running
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        ollama_running = result.returncode == 0
        
        # Check if our model exists
        model_exists = False
        if ollama_running and ollama_manager:
            model_exists = ollama_manager.model_exists()
        
        return jsonify({
            'ollama_running': ollama_running,
            'model_exists': model_exists,
            'model_name': ollama_manager.model_name if ollama_manager else 'N/A',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'ollama_running': False,
            'model_exists': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/examples')
def examples():
    """Page with example schemas and queries"""
    return render_template('examples.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Check if templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        logger.info("Created templates directory")
    
    if not os.path.exists('static'):
        os.makedirs('static')
        logger.info("Created static directory")
    
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000)
