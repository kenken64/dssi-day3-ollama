# Text-to-SQL Web Application

A modern, user-friendly web interface for the Text-to-SQL model powered by Flask and Ollama.

## 🌟 Features

- **Beautiful Interface**: Modern, responsive web design
- **Real-time Status**: Live monitoring of Ollama service and model availability
- **Interactive Examples**: Pre-built schemas and queries for different domains
- **SQL Validation**: Automatic syntax checking and error detection
- **Copy to Clipboard**: Easy result sharing and copying
- **Mobile Friendly**: Works perfectly on desktop, tablet, and mobile devices
- **Error Handling**: User-friendly error messages and troubleshooting

## 🚀 Quick Start

### 1. Start the Web Application
```bash
# Easy startup (auto-installs dependencies) - default port 5000
python start_webapp.py

# With custom port
python start_webapp.py --port 8080

# With custom host and port
python start_webapp.py --host 127.0.0.1 --port 3000

# With debug mode and custom model
python start_webapp.py --port 8080 --debug --model-name my-sql-model

# Or run directly with custom options
pip install flask>=2.3.0 requests>=2.31.0
python app.py --port 8080 --host 0.0.0.0 --debug
```

**Available Options:**
- `--port, -p`: Port number (default: 5000)
- `--host`: Host address (default: 0.0.0.0)
- `--debug`: Enable debug mode
- `--model-name`: Ollama model name (default: text-to-sql)

### 2. Access the Interface
Open your browser and navigate to: **http://localhost:PORT** (where PORT is your chosen port, default 5000)

Examples:
- Default: http://localhost:5000
- Custom port: http://localhost:8080
- Custom host: http://127.0.0.1:3000

### 3. Use the Application
1. **Enter Database Schema**: Paste your CREATE TABLE statements
2. **Write Natural Query**: Describe what you want in plain English
3. **Generate SQL**: Click the generate button
4. **Copy Results**: Use the copy button to get the SQL code

## 📱 Interface Overview

### Main Page (`/`)
- **Schema Input**: Large text area for database schema
- **Query Input**: Text area for natural language query
- **Status Indicator**: Shows Ollama and model status
- **Generate Button**: Processes your request
- **Results Section**: Shows generated SQL with validation

### Examples Page (`/examples`)
- **Pre-built Examples**: 5 different complexity levels
- **Domain Coverage**: E-commerce, healthcare, inventory, etc.
- **One-click Loading**: Click "Use This Example" to load
- **Complexity Badges**: Basic, Intermediate, Advanced levels

## 🔧 API Endpoints

### Status Check
```bash
# Default port
curl http://localhost:5000/api/status

# Custom port
curl http://localhost:8080/api/status
```

### Generate SQL
```bash
# Default port
curl -X POST http://localhost:5000/api/generate-sql \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE users (id INT, name VARCHAR(50));",
    "query": "Find all users"
  }'

# Custom port
curl -X POST http://localhost:8080/api/generate-sql \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE users (id INT, name VARCHAR(50));",
    "query": "Find all users"
  }'
```

### Validate SQL
```bash
# Default port
curl -X POST http://localhost:5000/api/validate-sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM users;"}'

# Custom port  
curl -X POST http://localhost:8080/api/validate-sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM users;"}'
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the web app image
docker build -f Dockerfile.webapp -t text-to-sql-webapp .

# Run the container
docker run -p 5000:5000 text-to-sql-webapp
```

### Docker Compose (with Ollama)
```yaml
version: '3.8'
services:
  webapp:
    build:
      context: .
      dockerfile: Dockerfile.webapp
    ports:
      - "5000:5000"
    depends_on:
      - ollama
    environment:
      - OLLAMA_HOST=ollama:11434
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

## 🧪 Testing the Web App

### Automated Demo
```bash
# Run API demo with default port (requires web app to be running)
python demo_webapp.py

# Run API demo with custom port
python demo_webapp.py --port 8080

# Run API demo with custom host and port
python demo_webapp.py --host 127.0.0.1 --port 3000
```

### Manual Testing
1. Visit http://localhost:PORT (replace PORT with your chosen port)
2. Try the example schemas and queries
3. Check status indicators
4. Verify SQL generation and validation

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `FLASK_DEBUG`: Set to `False` for production
- `FLASK_PORT`: Default port number (can be overridden by --port)
- `FLASK_HOST`: Default host address (can be overridden by --host)
- `OLLAMA_MODEL_NAME`: Default model name (can be overridden by --model-name)
- `SECRET_KEY`: Change the secret key for production use

### Customization
- **Templates**: Modify HTML templates in `templates/` directory
- **Styling**: Update CSS in the template files
- **API Endpoints**: Add new routes in `app.py`

## 🚨 Troubleshooting

### Web App Won't Start
```bash
# Check if Flask is installed
pip list | grep Flask

# Install if missing
pip install flask>=2.3.0
```

### "Service Unavailable" Status
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Check if model exists
ollama list | grep text-to-sql

# Deploy model if missing
python deployment_script.py
```

### API Returns Errors
- Verify JSON format in requests
- Check request headers include `Content-Type: application/json`
- Ensure required fields (`schema`, `query`) are provided
- Check server logs for detailed error messages

### Token Generation Loops
If you see repeating `{ end }<|end|>` patterns:
```bash
# Run the fix script
python fix_ollama_tokens.py

# Or recreate the model
python deployment_script.py --force
```

## 📊 Example Usage

### Basic Query
```json
{
  "schema": "CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL(10,2));",
  "query": "Find all products under $50"
}
```

### Complex Query
```json
{
  "schema": "CREATE TABLE customers (id INT, name VARCHAR(100)); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL(10,2));",
  "query": "Show top 5 customers by total order value"
}
```

## 🎯 Production Deployment

### Security Considerations
1. Change the Flask secret key
2. Set `FLASK_ENV=production`
3. Use a reverse proxy (nginx)
4. Enable HTTPS
5. Implement rate limiting
6. Add authentication if needed

### Performance Optimization
1. Use a production WSGI server (gunicorn)
2. Enable caching for static assets
3. Optimize database connections
4. Monitor resource usage
5. Scale horizontally if needed

The web application provides an intuitive interface for anyone to use the text-to-SQL model without command-line knowledge!
