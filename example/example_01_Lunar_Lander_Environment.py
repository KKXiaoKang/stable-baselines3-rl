import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


TRAIN_MODEL_FLAG = True

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Instantiate the agent
model = DQN("MlpPolicy", env, 
            verbose=1,
            buffer_size=200_000,  # 增大经验回放缓冲区
            learning_starts=10_000,  # 延迟学习开始
            batch_size=128,  # 增大批处理大小
            target_update_interval=500,  # 调整目标网络更新频率
            gamma=0.99,  # 调整折扣因子
            tensorboard_log="./dqn_lunar_log/")

# Train the agent and display a progress bar
if TRAIN_MODEL_FLAG:
    model.learn(total_timesteps=int(5e5), 
               progress_bar=True,
               log_interval=10)
    # Save the agent
    model.save("dqn_lunar")
    del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")