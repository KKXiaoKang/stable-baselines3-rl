#!/usr/bin/env python3
"""
Independent script to load and test a trained PPO model on RLKuavoGymEnv.
This script allows you to test a previously trained model without re-training.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path to import the environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from ppo_main import create_vec_env, test_model
from training_config import TESTING_CONFIG

def load_trained_model(model_path: str):
    """
    Load a trained PPO model from the specified path.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded PPO model
    """
    try:
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def test_loaded_model(model_path: str = None, n_episodes: int = None, max_steps: int = None):
    """
    Load and test a trained model.
    
    Args:
        model_path: Path to the model file (default: latest model in model_dir)
        n_episodes: Number of test episodes (default: from config)
        max_steps: Maximum steps per episode (default: from config)
    """
    print("=" * 60)
    print("Testing Trained PPO Model on RLKuavoGymEnv")
    print("=" * 60)
    
    # Use default model path if not specified
    if model_path is None:
        from training_config import LOGGING_CONFIG
        model_dir = LOGGING_CONFIG["model_dir"]
        model_path = os.path.join(model_dir, "ppo_kuavo_final")
        print(f"Using default model path: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file not found at {model_path}.zip")
        print("Available models in model directory:")
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    print(f"  - {os.path.join(model_dir, file)}")
        return
    
    try:
        # Load the trained model
        model = load_trained_model(model_path)
        
        # Create environment
        print("\nCreating environment...")
        env = create_vec_env()
        print("Environment created successfully!")
        
        # Test the model
        print("\nStarting model testing...")
        test_model(model, env, n_episodes=n_episodes)
        
        print("\n" + "=" * 60)
        print("Testing completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        print("\nTest script finished.")

def list_available_models():
    """List all available trained models."""
    from training_config import LOGGING_CONFIG
    model_dir = LOGGING_CONFIG["model_dir"]
    
    print("Available trained models:")
    if os.path.exists(model_dir):
        models = []
        for file in os.listdir(model_dir):
            if file.endswith('.zip'):
                model_path = os.path.join(model_dir, file)
                # Get file size and modification time
                stat = os.stat(model_path)
                size_mb = stat.st_size / (1024 * 1024)
                models.append((file, model_path, size_mb, stat.st_mtime))
        
        if models:
            # Sort by modification time (newest first)
            models.sort(key=lambda x: x[3], reverse=True)
            for file, path, size, mtime in models:
                import time
                mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f"  - {file} ({size:.1f} MB, {mtime_str})")
                print(f"    Path: {path}")
        else:
            print("  No trained models found.")
    else:
        print(f"  Model directory does not exist: {model_dir}")

def main():
    """Main function with command line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a trained PPO model on RLKuavoGymEnv")
    parser.add_argument("--model", "-m", type=str, help="Path to the model file")
    parser.add_argument("--episodes", "-e", type=int, help="Number of test episodes")
    parser.add_argument("--max-steps", "-s", type=int, help="Maximum steps per episode")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    # Use command line arguments or defaults
    model_path = args.model
    n_episodes = args.episodes if args.episodes is not None else TESTING_CONFIG["n_test_episodes"]
    max_steps = args.max_steps if args.max_steps is not None else TESTING_CONFIG["max_test_steps"]
    
    test_loaded_model(model_path, n_episodes, max_steps)

if __name__ == "__main__":
    main()
