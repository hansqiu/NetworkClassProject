import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from net_env import NetworkEnv  
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning) 

ray.init()

def env_creator(env_config):
    return NetworkEnv() 

register_env('netenv-v0', env_creator)

config = (PPOConfig()
          .training(gamma=0.999, lr=0.001)
          .environment(env='netenv-v0')
          .resources(num_gpus=0)
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
         )

algo = config.build()
network_wide_utilizations = []
episode_rewards = []

for _ in range(10):
    result = algo.train()
    episode_rewards_mean = result['episode_reward_mean']
    episode_rewards.append(episode_rewards_mean)

    env = algo.workers.local_worker().env
    if hasattr(env, 'get_episode_utilization'):
        temp = env.get_episode_utilization()
        network_wide_utilizations.append(temp)

average_utilizations_per_episode = [np.mean(episode) for episode in network_wide_utilizations]
average_network_utilization = np.mean(network_wide_utilizations)*10
print(average_network_utilization)

if network_wide_utilizations:
    episodes = np.arange(len(network_wide_utilizations))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, average_utilizations_per_episode, label='Network-wide Utilization')
    plt.xlabel('Episode')
    plt.ylabel('Average Utilization')
    plt.title('Network-wide Utilization Over Episodes')
    plt.legend()
    plt.show()


