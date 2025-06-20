#!/bin/bash

# Text-to-SQL Ollama Model - Complete Installation Script
# This script automates the entire setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="text-to-sql-ollama"
PYTHON_VERSION="3.10"
VENV_NAME="text-to-sql-env"
DEFAULT_MODEL="defog/sqlcoder-7b"
SAMPLES_QUICK=1000
SAMPLES_FULL=10000

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "  Text-to-SQL Ollama Model Installation Script"
    echo "=================================================="
    echo -e "${NC}"
}

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$(printf '%s\n' "3.8" "$PYTHON_VER" | sort -V | head -n1)" == "3.8" ]]; then
            log_success "Python $PYTHON_VER found"
        else
            log_error "Python 3.8+ required, found $PYTHON_VER"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 not found. Please install pip"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_error "git not found. Please install git"
        exit 1
    fi
    
    # Check available disk space (need at least 20GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=20971520  # 20GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        log_warning "Low disk space. At least 20GB recommended"
    fi
    
    # Check CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read name memory; do
            log_info "GPU: $name ($memory MB VRAM)"
        done
    else
        log_warning "No NVIDIA GPU detected. Training will use CPU (slower)"
    fi
    
    log_success "System requirements check completed"
}

install_ollama() {
    log_info "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        log_success "Ollama already installed"
        return
    fi
    
    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            log_error "Homebrew not found. Please install Ollama manually from https://ollama.ai/"
            exit 1
        fi
    else
        log_error "Unsupported OS. Please install Ollama manually from https://ollama.ai/"
        exit 1
    fi
    
    # Start Ollama service
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start ollama
        sudo systemctl enable ollama
    fi
    
    log_success "Ollama installed successfully"
}

create_virtual_environment() {
    log_info "Creating Python virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        log_warning "Virtual environment already exists. Removing old one..."
        rm -rf "$VENV_NAME"
    fi
    
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created: $VENV_NAME"
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Install PyTorch first (with CUDA support if available)
    if command -v nvidia-smi &> /dev/null; then
        log_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    pip install -r requirements.txt
    
    log_success "Dependencies installed successfully"
}

setup_project_structure() {
    log_info "Setting up project structure..."
    
    # Create necessary directories
    mkdir -p {models,cache,logs,outputs,processed_data,examples,tests,configs}
    
    # Create __init__.py files
    touch {models,examples,tests,configs}/__init__.py
    
    # Make scripts executable
    chmod +x *.py
    
    log_success "Project structure created"
}

download_and_prepare_data() {
    log_info "Downloading and preparing dataset..."
    
    source "$VENV_NAME/bin/activate"
    
    # Download dataset to cache
    python -c "
from datasets import load_dataset
import os
print('Downloading Gretel AI synthetic text-to-SQL dataset...')
dataset = load_dataset('gretelai/synthetic_text_to_sql', cache_dir='./cache')
print(f'Dataset downloaded: {len(dataset[\"train\"])} samples')
"
    
    log_success "Dataset downloaded and cached"
}

train_model() {
    local mode="$1"
    local samples="$2"
    
    log_info "Training model in $mode mode with $samples samples..."
    
    source "$VENV_NAME/bin/activate"
    
    # Run training
    python text_to_sql_train.py \
        --mode "$mode" \
        --max-samples "$samples" \
        --base-model "$DEFAULT_MODEL"
    
    log_success "Model training completed"
}

deploy_model() {
    log_info "Deploying model to Ollama..."
    
    source "$VENV_NAME/bin/activate"
    
    # Deploy model
    python deploy.py --config config.yaml --force
    
    log_success "Model deployed to Ollama"
}

test_deployment() {
    log_info "Testing model deployment..."
    
    source "$VENV_NAME/bin/activate"
    
    # Run tests
    python test.py --config config.yaml --quick
    
    log_success "Deployment testing completed"
}

create_desktop_shortcuts() {
    log_info "Creating desktop shortcuts..."
    
    # Create activation script
    cat > activate_env.sh << EOF
#!/bin/bash
cd "$(pwd)"
source $VENV_NAME/bin/activate
exec "\$@"
EOF
    chmod +x activate_env.sh
    
    # Create quick start script
    cat > quick_start.sh << EOF
#!/bin/bash
cd "$(pwd)"
source $VENV_NAME/bin/activate
python examples/quick_start.py
EOF
    chmod +x quick_start.sh
    
    # Create training script
    cat > retrain_model.sh << EOF
#!/bin/bash
cd "$(pwd)"
source $VENV_NAME/bin/activate
python text_to_sql_train.py --mode full --max-samples $SAMPLES_FULL
EOF
    chmod +x retrain_model.sh
    
    log_success "Desktop shortcuts created"
}

show_usage_instructions() {
    echo
    log_success "Installation completed successfully!"
    echo
    echo -e "${GREEN}Next Steps:${NC}"
    echo "1. Activate the environment:"
    echo "   source $VENV_NAME/bin/activate"
    echo
    echo "2. Run quick examples:"
    echo "   python examples/quick_start.py"
    echo
    echo "3. Test the model:"
    echo "   ollama run text-to-sql \"CREATE TABLE users (id INT, name VARCHAR(50)); Find all users\""
    echo
    echo "4. Run comprehensive tests:"
    echo "   python test.py --config config.yaml"
    echo
    echo "5. Retrain with more data:"
    echo "   python text_to_sql_train.py --mode full --max-samples $SAMPLES_FULL"
    echo
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "- View available models: ollama list"
    echo "- Interactive demo: python deploy.py --demo"
    echo "- Performance testing: python test.py --performance-only"
    echo "- Check logs: tail -f logs/*.log"
    echo
    echo -e "${YELLOW}Configuration:${NC}"
    echo "- Config file: config.yaml"
    echo "- Model location: models/"
    echo "- Cache location: cache/"
    echo "- Logs location: logs/"
}

cleanup_on_error() {
    log_error "Installation failed. Cleaning up..."
    
    # Remove virtual environment if it exists
    if [ -d "$VENV_NAME" ]; then
        rm -rf "$VENV_NAME"
    fi
    
    # Remove any partially created directories
    rm -rf models cache logs outputs processed_data
    
    exit 1
}

main() {
    # Handle interruption
    trap cleanup_on_error INT TERM ERR
    
    print_banner
    
    # Parse command line arguments
    MODE="quick"  # default mode
    SAMPLES="$SAMPLES_QUICK"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                MODE="full"
                SAMPLES="$SAMPLES_FULL"
                shift
                ;;
            --quick)
                MODE="quick"
                SAMPLES="$SAMPLES_QUICK"
                shift
                ;;
            --samples)
                SAMPLES="$2"
                shift 2
                ;;
            --model)
                DEFAULT_MODEL="$2"
                shift 2
                ;;
            --no-training)
                SKIP_TRAINING=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --full              Full installation with complete dataset"
                echo "  --quick             Quick installation with limited dataset (default)"
                echo "  --samples N         Use N samples for training"
                echo "  --model MODEL       Use specific base model"
                echo "  --no-training       Skip model training"
                echo "  --help              Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "Starting installation in $MODE mode with $SAMPLES samples"
    
    # Installation steps
    check_system_requirements
    install_ollama
    create_virtual_environment
    install_dependencies
    setup_project_structure
    download_and_prepare_data
    
    if [ "$SKIP_TRAINING" != "true" ]; then
        train_model "$MODE" "$SAMPLES"
        deploy_model
        test_deployment
    else
        log_info "Skipping model training as requested"
    fi
    
    create_desktop_shortcuts
    show_usage_instructions
    
    log_success "Installation completed successfully!"
}

# Script options help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Text-to-SQL Ollama Model Installation Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Installation Modes:"
    echo "  --quick             Quick setup with 1K samples (default, ~10 minutes)"
    echo "  --full              Full setup with 10K samples (~2-4 hours)"
    echo
    echo "Options:"
    echo "  --samples N         Use N samples for training"
    echo "  --model MODEL       Specify base model (default: defog/sqlcoder-7b)"
    echo "  --no-training       Skip training, only setup environment"
    echo "  --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0                  # Quick installation"
    echo "  $0 --full           # Full installation"
    echo "  $0 --samples 5000   # Custom sample count"
    echo "  $0 --model codellama/CodeLlama-7b-Instruct-hf"
    echo
    echo "Requirements:"
    echo "  - Python 3.8+"
    echo "  - 20GB+ disk space"
    echo "  - Internet connection"
    echo "  - NVIDIA GPU (optional, recommended)"
    exit 0
fi

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root is not recommended. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start main installation
main "$@"
