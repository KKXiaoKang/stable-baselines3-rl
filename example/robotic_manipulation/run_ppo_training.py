#!/usr/bin/env python3
"""
Simple script to run PPO training on RLKuavoGymEnv.
This script provides a quick way to start training with your custom environment.
"""

import sys
import os

# Add the current directory to Python path to import the environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo_main import train_ppo, test_model

def main():
    """Main function to run PPO training."""
    print("=" * 60)
    print("PPO Training for RLKuavoGymEnv")
    print("=" * 60)
    
    try:
        # Train the model
        print("\n1. Starting PPO training...")
        model, env = train_ppo()
        
        # Test the trained model
        print("\n2. Testing the trained model...")
        test_model(model, env, n_episodes=3)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        print("\nScript finished.")

if __name__ == "__main__":
    main()
