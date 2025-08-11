# Quick Start Guide - PPO Training for RLKuavoGymEnv

## Prerequisites

1. **ROS Setup**: Make sure ROS is installed and `roscore` is running
2. **Isaac Lab**: Ensure your Isaac Lab simulation is running
3. **Python Dependencies**: Install required packages:
   ```bash
   pip install stable-baselines3 gymnasium tensorboard
   ```

## Quick Start Options

### Option 1: Automated Training (Recommended)

Use the automated script that handles everything:

```bash
cd example/robotic_manipulation
chmod +x start_training.sh
./start_training.sh
```

This script will:
- ✅ Check dependencies
- ✅ Verify ROS setup
- ✅ Test your environment
- ✅ Start TensorBoard monitoring
- ✅ Begin PPO training

### Option 2: Manual Step-by-Step

1. **Test your environment first**:
   ```bash
   python test_env.py
   ```

2. **Start training**:
   ```bash
   python run_ppo_training.py
   ```

3. **Monitor with TensorBoard** (in another terminal):
   ```bash
   tensorboard --logdir ppo_kuavo_logs
   ```

### Option 3: Custom Configuration

1. **Edit training parameters** in `training_config.py`
2. **Run training**:
   ```bash
   python ppo_main.py
   ```

## What You'll Get

After training, you'll have:

- 📁 `ppo_kuavo_logs/` - Training logs and TensorBoard data
- 📁 `ppo_kuavo_models/` - Saved model checkpoints
- 📊 Training metrics and progress

## Monitoring Training

Open your browser to `http://localhost:6006` to see:
- Training rewards over time
- Loss curves
- Episode statistics
- Environment metrics

## Common Issues & Solutions

### Issue: "ROS node already exists"
**Solution**: Ensure only one environment instance is running

### Issue: "Service not available"
**Solution**: Check that Isaac Lab simulation is running

### Issue: "Import error"
**Solution**: Install missing dependencies:
```bash
pip install stable_baselines3 gymnasium tensorboard
```

### Issue: "Permission denied"
**Solution**: Make script executable:
```bash
chmod +x start_training.sh
```

## Next Steps

1. **Experiment with parameters** in `training_config.py`
2. **Try different reward functions** in your environment
3. **Extend the environment** with new tasks
4. **Use trained models** for deployment

## Need Help?

- Check the full `README.md` for detailed documentation
- Review `training_config.py` for all available options
- Use `test_env.py` to debug environment issues
