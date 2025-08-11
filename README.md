# PPO Training for RLKuavoGymEnv

This directory contains scripts for training PPO policies on the `RLKuavoGymEnv` environment.

## Files Overview

- `rl_kuavo_gym_env.py` - Your custom Gymnasium environment for the Kuavo robot
- `ppo_main.py` - Main PPO training script with comprehensive functionality
- `run_ppo_training.py` - Simple script to run PPO training
- `training_config.py` - Configuration file for easy parameter adjustment
- `README.md` - This file

## Quick Start

### 1. Basic Training

To start training immediately with default settings:

```bash
cd example/robotic_manipulation
python run_ppo_training.py
```

### 2. Customized Training

To customize training parameters, edit `training_config.py` and then run:

```bash
python ppo_main.py
```

## Configuration

### Environment Configuration (`ENV_CONFIG`)

Key parameters you can adjust in `training_config.py`:

- `debug`: Enable debug mode for detailed logging
- `wbc_observation_enabled`: Enable WBC observations (True/False)
- `image_size`: Image observation size (default: 224x224)
- `enable_roll_pitch_control`: Enable roll/pitch control for base

### Training Configuration (`TRAINING_CONFIG`)

- `total_timesteps`: Total training timesteps (default: 100,000)
- `n_envs`: Number of parallel environments (default: 1 for ROS)
- `random_seed`: Random seed for reproducibility

### PPO Hyperparameters (`PPO_CONFIG`)

- `learning_rate`: Learning rate (default: 3e-4)
- `n_steps`: Steps per update (default: 2048)
- `batch_size`: Batch size (default: 64)
- `n_epochs`: Epochs per update (default: 10)
- `gamma`: Discount factor (default: 0.99)

### Environment Flags (`ENV_FLAGS`)

These flags control the behavior of your environment:

- `TEST_DEMO_USE_ACTION_16_DIM`: Use 16-dim joint control (False = 6-dim position control)
- `USE_CMD_VEL`: Use velocity control
- `LEARN_TARGET_EEF_POSE_TARGET`: Learn target end-effector poses

## Training Output

The training will create:

- `ppo_kuavo_logs/` - TensorBoard logs and training metrics
- `ppo_kuavo_models/` - Saved model checkpoints and final model

### Monitoring Training

To monitor training progress with TensorBoard:

```bash
tensorboard --logdir ppo_kuavo_logs
```

Then open your browser to `http://localhost:6006`

## Usage Examples

### Example 1: Quick Training with Default Settings

```python
from ppo_main import train_ppo, test_model

# Train the model
model, env = train_ppo()

# Test the trained model
test_model(model, env, n_episodes=3)
```

### Example 2: Custom Environment Configuration

```python
from training_config import ENV_CONFIG
from ppo_main import create_vec_env, train_ppo

# Modify environment config
ENV_CONFIG["debug"] = True
ENV_CONFIG["wbc_observation_enabled"] = True

# Create environment and train
env = create_vec_env(n_envs=1, debug=True)
model, env = train_ppo()
```

### Example 3: Load and Test a Trained Model

```python
from stable_baselines3 import PPO
from ppo_main import create_vec_env, test_model

# Load trained model
model = PPO.load("ppo_kuavo_models/ppo_kuavo_final")

# Create environment
env = create_vec_env()

# Test the model
test_model(model, env, n_episodes=5)
```

## Important Notes

### ROS Environment Considerations

1. **Single Environment**: ROS environments work best with `n_envs=1` due to node conflicts
2. **Node Initialization**: The environment handles ROS node initialization automatically
3. **Resource Management**: Ensure proper cleanup with `env.close()`

### Training Tips

1. **Start Small**: Begin with fewer timesteps (e.g., 10,000) to test your setup
2. **Monitor Logs**: Use TensorBoard to monitor training progress
3. **Checkpoint Regularly**: Models are saved every 10,000 steps by default
4. **Debug Mode**: Enable debug mode to see detailed environment information

### Common Issues

1. **ROS Node Conflicts**: If you get ROS node errors, ensure only one environment instance
2. **Memory Issues**: Reduce batch size or number of environments if you run out of memory
3. **Training Instability**: Try reducing learning rate or increasing entropy coefficient

## Environment Details

Your `RLKuavoGymEnv` provides:

- **Observation Space**: Dict with pixels, agent_pos, and environment_state
- **Action Space**: Box with normalized actions [-1, 1]
- **Reward**: Based on end-effector position or joint angle targets
- **Termination**: Episode timeout or task completion

The environment supports both:
- **Position Control**: 6-dimensional end-effector position control
- **Joint Control**: 16-dimensional joint angle control

## Next Steps

1. **Experiment with Parameters**: Try different hyperparameters in `training_config.py`
2. **Add Custom Callbacks**: Implement custom callbacks for specific monitoring needs
3. **Extend Environment**: Add more complex tasks or observations to your environment
4. **Multi-Environment Training**: Experiment with multiple environments if ROS setup allows
