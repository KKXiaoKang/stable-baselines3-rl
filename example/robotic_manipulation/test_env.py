#!/usr/bin/env python3
"""
Test script for RLKuavoGymEnv to verify it works correctly before training.
This script performs basic environment tests without starting PPO training.
"""

import sys
import os
import time
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_kuavo_gym_env import RLKuavoGymEnv

def test_environment_basic():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing RLKuavoGymEnv Basic Functionality")
    print("=" * 60)
    
    try:
        # Create environment with debug mode
        print("Creating environment...")
        env = RLKuavoGymEnv(
            debug=True,
            image_size=(224, 224),
            enable_roll_pitch_control=False,
            vel_smoothing_factor=0.3,
            arm_smoothing_factor=0.4,
            wbc_observation_enabled=False,
            action_dim=None,
            image_obs=True,
            render_mode=None,
            use_gripper=True,
            gripper_penalty=0.0
        )
        
        print("✓ Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset
        print("\nTesting reset...")
        obs, info = env.reset()
        print("✓ Reset successful!")
        print(f"Observation shape: {obs.shape}")
        print(f"Agent_pos dim: {env.agent_dim}")
        print(f"Environment_state dim: {env.env_state_dim}")
        
        # Test a few steps with random actions
        print("\nTesting random actions...")
        for step in range(5):
            print(f"\nStep {step + 1}/5:")
            
            # Sample random action
            action = env.action_space.sample()
            print(f"  Action shape: {action.shape}")
            print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"  Reward: {reward:.4f}")
            print(f"  Done: {done}")
            print(f"  Truncated: {truncated}")
            print(f"  Info: {info}")
            
            if done:
                print("  Episode ended, resetting...")
                obs, info = env.reset()
                break
            
            # Small delay to allow ROS processing
            time.sleep(0.1)
        
        print("\n✓ Basic environment test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during environment test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'env' in locals():
            env.close()
            print("Environment closed.")

def test_environment_episode():
    """Test a complete episode."""
    print("\n" + "=" * 60)
    print("Testing Complete Episode")
    print("=" * 60)
    
    try:
        # Create environment
        env = RLKuavoGymEnv(debug=False)  # Disable debug for cleaner output
        
        # Run one complete episode
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 200  # Limit steps for testing
        
        print(f"Starting episode (max {max_steps} steps)...")
        
        while step_count < max_steps:
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress every 10 steps
            if step_count % 10 == 0:
                print(f"  Step {step_count}: Reward so far: {episode_reward:.3f}")
            
            if done:
                print(f"  Episode ended at step {step_count}")
                break
        
        print(f"\nEpisode completed:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {episode_reward:.3f}")
        print("✓ Episode test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during episode test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'env' in locals():
            env.close()

def main():
    """Main test function."""
    print("RLKuavoGymEnv Test Suite")
    print("This script tests your environment before starting PPO training.")
    print("Make sure your ROS setup is running and Isaac Lab is ready.")
    print()
    
    # Wait for user confirmation
    input("Press Enter to start testing (ensure ROS and Isaac Lab are running)...")
    
    # Run tests
    basic_test_passed = test_environment_basic()
    episode_test_passed = test_environment_episode()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Basic functionality test: {'✓ PASSED' if basic_test_passed else '✗ FAILED'}")
    print(f"Episode test: {'✓ PASSED' if episode_test_passed else '✗ FAILED'}")
    
    if basic_test_passed and episode_test_passed:
        print("\n🎉 All tests passed! Your environment is ready for training.")
        print("You can now run: python run_ppo_training.py")
    else:
        print("\n❌ Some tests failed. Please check your ROS setup and environment configuration.")
        print("Common issues:")
        print("1. ROS master not running (run 'roscore')")
        print("2. Isaac Lab simulation not started")
        print("3. Required ROS topics not available")
        print("4. Network configuration issues")

if __name__ == "__main__":
    main()
