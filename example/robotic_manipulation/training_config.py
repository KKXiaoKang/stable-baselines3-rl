"""
Configuration file for PPO training on RLKuavoGymEnv.
Modify these parameters to adjust the training behavior.
"""

# Environment Configuration
ENV_CONFIG = {
    "debug": False,                    # Enable debug mode for environment
    "image_size": (224, 224),         # Image observation size
    "enable_roll_pitch_control": False, # Enable roll/pitch control for base
    "vel_smoothing_factor": 0.2,      # Velocity smoothing factor
    "arm_smoothing_factor": 0.2,      # Arm action smoothing factor
    "wbc_observation_enabled": False, # Enable WBC observations (True/False)
    "action_dim": None,               # Action dimension (None for auto)
    "image_obs": True,                # Include image observations
    "render_mode": None,              # Render mode
    "use_gripper": True,              # Use gripper control
    "gripper_penalty": 0.0,           # Gripper penalty coefficient
}

# Training Configuration
TRAINING_CONFIG = {
    "total_timesteps": 1000000,       # Total training timesteps - 1M 
    "n_envs": 1,                      # Number of parallel environments
    "random_seed": 42,                # Random seed for reproducibility
}

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,            # Learning rate
    "n_steps": 2000,                  # Number of steps per update
    "batch_size": 256,                 # Batch size
    "n_epochs": 10,                   # Number of epochs per update
    "gamma": 0.99,                    # Discount factor
    "gae_lambda": 0.95,               # GAE lambda parameter
    "clip_range": 0.2,                # PPO clip range
    "clip_range_vf": None,            # Value function clip range
    "normalize_advantage": True,      # Normalize advantage estimates
    "ent_coef": 0.01,                # Entropy coefficient
    "vf_coef": 0.5,                  # Value function coefficient
    "max_grad_norm": 0.5,            # Maximum gradient norm
    "use_sde": False,                # Use state-dependent exploration
    "sde_sample_freq": -1,           # SDE sampling frequency
    "target_kl": None,               # Target KL divergence
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),  # Network architecture (fixed format)
        "activation_fn": "relu",      # Activation function (will be converted to proper function)
    },
    "verbose": 1,                    # Verbosity level
}

# Logging and Saving Configuration
LOGGING_CONFIG = {
    "log_dir": "ppo_kuavo_logs",     # Directory for logs
    "model_dir": "ppo_kuavo_models", # Directory for saved models
    "checkpoint_freq": 10000,        # Save checkpoint every N steps
    "tensorboard_log": True,         # Enable TensorBoard logging
}

# Testing Configuration
TESTING_CONFIG = {
    "n_test_episodes": 3,            # Number of test episodes
    "max_test_steps": 200,           # Maximum steps per test episode
    "deterministic": True,           # Use deterministic actions for testing
}

# Environment-specific flags (from your environment)
# These should match the flags in your RLKuavoGymEnv
ENV_FLAGS = {
    "TEST_DEMO_USE_ACTION_16_DIM": False,  # Use 16-dim joint control
    "USE_CMD_VEL": False,                  # Use velocity control
    "IF_USE_ZERO_OBS_FLAG": False,         # Use zero observations
    "IF_USE_ARM_MPC_CONTROL": False,       # Use MPC control
    "LEARN_TARGET_EEF_POSE_TARGET": True,  # Learn target end-effector poses
}
