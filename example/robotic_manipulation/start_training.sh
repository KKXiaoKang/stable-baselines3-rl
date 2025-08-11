#!/bin/bash

# PPO Training Startup Script for RLKuavoGymEnv
# This script automates the training process with proper setup and monitoring

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if ROS is running
check_ros() {
    print_status "Checking ROS setup..."
    
    if ! command -v roscore &> /dev/null; then
        print_error "ROS is not installed or not in PATH"
        exit 1
    fi
    
    if ! pgrep -x "roscore" > /dev/null; then
        print_warning "roscore is not running. Please start it first:"
        echo "  roscore &"
        echo "  sleep 3"
        read -p "Press Enter after starting roscore, or Ctrl+C to cancel..."
    fi
    
    print_success "ROS setup verified"
}

# Function to check Python dependencies
check_dependencies() {
    print_status "Checking Python dependencies..."
    
    python3 -c "import stable_baselines3" 2>/dev/null || {
        print_error "stable_baselines3 not found. Please install it:"
        echo "  pip install stable-baselines3"
        exit 1
    }
    
    python3 -c "import gymnasium" 2>/dev/null || {
        print_error "gymnasium not found. Please install it:"
        echo "  pip install gymnasium"
        exit 1
    }
    
    python3 -c "import rospy" 2>/dev/null || {
        print_error "rospy not found. Please install ROS Python packages:"
        echo "  sudo apt-get install python3-rospy"
        exit 1
    }
    
    print_success "Python dependencies verified"
}

# Function to test environment
test_environment() {
    print_status "Testing environment..."
    
    if python3 test_env.py; then
        print_success "Environment test passed"
    else
        print_error "Environment test failed. Please check your setup."
        exit 1
    fi
}

# Function to start training
start_training() {
    print_status "Starting PPO training..."
    
    # Create logs directory
    mkdir -p ppo_kuavo_logs
    mkdir -p ppo_kuavo_models
    
    # Start training
    python3 run_ppo_training.py
    
    print_success "Training completed"
}

# Function to start TensorBoard
start_tensorboard() {
    print_status "Starting TensorBoard for monitoring..."
    print_status "TensorBoard will be available at: http://localhost:6006"
    
    # Start TensorBoard in background
    tensorboard --logdir ppo_kuavo_logs --port 6006 &
    TENSORBOARD_PID=$!
    
    echo $TENSORBOARD_PID > tensorboard.pid
    print_success "TensorBoard started with PID: $TENSORBOARD_PID"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Kill TensorBoard if running
    if [ -f tensorboard.pid ]; then
        TB_PID=$(cat tensorboard.pid)
        if kill -0 $TB_PID 2>/dev/null; then
            kill $TB_PID
            print_status "TensorBoard stopped"
        fi
        rm -f tensorboard.pid
    fi
}

# Set up trap to cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    echo "=========================================="
    echo "PPO Training for RLKuavoGymEnv"
    echo "=========================================="
    echo
    
    # Check if we're in the right directory
    if [ ! -f "rl_kuavo_gym_env.py" ]; then
        print_error "Please run this script from the robotic_manipulation directory"
        exit 1
    fi
    
    # Parse command line arguments
    SKIP_TESTS=false
    SKIP_TENSORBOARD=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-tensorboard)
                SKIP_TENSORBOARD=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-tests        Skip environment testing"
                echo "  --skip-tensorboard  Skip TensorBoard startup"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run checks
    check_dependencies
    check_ros
    
    # Test environment (unless skipped)
    if [ "$SKIP_TESTS" = false ]; then
        test_environment
    else
        print_warning "Skipping environment tests"
    fi
    
    # Start TensorBoard (unless skipped)
    if [ "$SKIP_TENSORBOARD" = false ]; then
        start_tensorboard
        sleep 2  # Give TensorBoard time to start
    else
        print_warning "Skipping TensorBoard startup"
    fi
    
    # Start training
    start_training
    
    print_success "All done! Check ppo_kuavo_models/ for saved models"
}

# Run main function
main "$@"
