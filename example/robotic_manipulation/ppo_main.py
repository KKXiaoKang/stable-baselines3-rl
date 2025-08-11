import gymnasium as gym
import numpy as np
import os
import time
import torch.nn as nn
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import your custom environment
from rl_kuavo_gym_env import RLKuavoGymEnv
from training_config import ENV_CONFIG, TRAINING_CONFIG, PPO_CONFIG, LOGGING_CONFIG, TESTING_CONFIG

def create_env(env_id: int = 0, debug: bool = None) -> RLKuavoGymEnv:
    """
    Create a single RLKuavoGymEnv instance.
    
    Args:
        env_id: Environment ID for logging purposes
        debug: Whether to enable debug mode
    
    Returns:
        RLKuavoGymEnv instance wrapped with Monitor
    """
    # Use debug from config if not provided
    if debug is None:
        debug = ENV_CONFIG["debug"]
    
    def _make_env():
        env = RLKuavoGymEnv(
            debug=debug,
            image_size=ENV_CONFIG["image_size"],
            enable_roll_pitch_control=ENV_CONFIG["enable_roll_pitch_control"],
            vel_smoothing_factor=ENV_CONFIG["vel_smoothing_factor"],
            arm_smoothing_factor=ENV_CONFIG["arm_smoothing_factor"],
            wbc_observation_enabled=ENV_CONFIG["wbc_observation_enabled"],
            action_dim=ENV_CONFIG["action_dim"],
            image_obs=ENV_CONFIG["image_obs"],
            render_mode=ENV_CONFIG["render_mode"],
            use_gripper=ENV_CONFIG["use_gripper"],
            gripper_penalty=ENV_CONFIG["gripper_penalty"]
        )
        # Wrap with Monitor for logging
        env = Monitor(env, info_keywords=("success",))
        return env
    
    return _make_env()

def create_vec_env(n_envs: int = None, debug: bool = None) -> DummyVecEnv:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        debug: Whether to enable debug mode
    
    Returns:
        Vectorized environment
    """
    # Use config values if not provided
    if n_envs is None:
        n_envs = TRAINING_CONFIG["n_envs"]
    if debug is None:
        debug = ENV_CONFIG["debug"]
        
    if n_envs == 1:
        # Single environment
        env = create_env(debug=debug)
        return DummyVecEnv([lambda: env])
    else:
        # Multiple environments - use SubprocVecEnv for true parallelism
        # Note: ROS environments might not work well with SubprocVecEnv due to node conflicts
        # For now, we'll use DummyVecEnv with multiple instances
        env_fns = [lambda i=i: create_env(env_id=i, debug=debug) for i in range(n_envs)]
        return DummyVecEnv(env_fns)

def train_ppo():
    """Main training function for PPO on RLKuavoGymEnv."""
    
    # Set random seed for reproducibility
    set_random_seed(TRAINING_CONFIG["random_seed"])
    
    # Create logs directory
    log_dir = LOGGING_CONFIG["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    # Create model directory
    model_dir = LOGGING_CONFIG["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    
    print("Creating RLKuavoGymEnv environment...")
    
    # Create vectorized environment
    # Note: For ROS environments, it's recommended to start with 1 environment
    # Multiple environments might cause ROS node conflicts
    env = create_vec_env()
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # PPO hyperparameters from config
    ppo_params = PPO_CONFIG.copy()
    ppo_params["tensorboard_log"] = log_dir
    
    # Convert activation function string to proper function
    if "policy_kwargs" in ppo_params and "activation_fn" in ppo_params["policy_kwargs"]:
        activation_str = ppo_params["policy_kwargs"]["activation_fn"]
        if activation_str == "relu":
            ppo_params["policy_kwargs"]["activation_fn"] = nn.ReLU
        elif activation_str == "tanh":
            ppo_params["policy_kwargs"]["activation_fn"] = nn.Tanh
        elif activation_str == "sigmoid":
            ppo_params["policy_kwargs"]["activation_fn"] = nn.Sigmoid
        else:
            # Default to ReLU if unknown activation function
            print(f"Warning: Unknown activation function '{activation_str}', using ReLU")
            ppo_params["policy_kwargs"]["activation_fn"] = nn.ReLU
    
    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, **ppo_params)
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback - save model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=LOGGING_CONFIG["checkpoint_freq"],
        save_path=model_dir,
        name_prefix="ppo_kuavo",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (optional - requires separate eval environment)
    # eval_env = create_vec_env(n_envs=1, debug=False)
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=f"{model_dir}/best_model",
    #     log_path=log_dir,
    #     eval_freq=5000,
    #     deterministic=True,
    #     render=False,
    # )
    # callbacks.append(eval_callback)
    
    print("Starting PPO training...")
    print(f"Training for {TRAINING_CONFIG['total_timesteps']} timesteps...")
    
    # Start training
    start_time = time.time()
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "ppo_kuavo_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, env

def test_model(model, env, n_episodes: int = None):
    """Test the trained model."""
    # Use config values if not provided
    if n_episodes is None:
        n_episodes = TESTING_CONFIG["n_test_episodes"]
    max_steps = TESTING_CONFIG["max_test_steps"]
    
    print(f"\nTesting model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        obs = env.reset()
        episode_reward = 0.0  # Initialize as float
        step_count = 0
        
        while step_count < max_steps:
            # Get action from model
            action, _states = model.predict(obs, deterministic=TESTING_CONFIG["deterministic"])
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            # Handle vectorized environment - extract scalar values
            if isinstance(reward, np.ndarray):
                reward = reward[0]  # Take first environment's reward
            if isinstance(done, np.ndarray):
                done = done[0]  # Take first environment's done flag
            if isinstance(info, list):
                info = info[0]  # Take first environment's info
            
            episode_reward += reward
            step_count += 1
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Reward so far: {episode_reward:.3f}")
            
            if done:
                break
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {episode_reward:.3f}")
        if 'success' in info:
            print(f"  Success: {info['success']}")

if __name__ == "__main__":
    try:
        # Train the model
        model, env = train_ppo()
        
        # Test the trained model
        test_model(model, env, n_episodes=3)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        print("\nTraining script finished.")