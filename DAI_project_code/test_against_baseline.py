import gym
import slimevolleygym
from dueling_ddqn import Agent
import numpy as np

action_table = [[0, 0, 0],  # NOOP
                [1, 0, 0],  # LEFT (forward)
                [1, 0, 1],  # UPLEFT (forward jump)
                [0, 0, 1],  # UP (jump)
                [0, 1, 1],  # UPRIGHT (backward jump)
                [0, 1, 0]]  # RIGHT (backward)

env = gym.make("SlimeVolley-v0")


agent_right = Agent(gamma=0.99, epsilon=0, lr=5e-4,
                    input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
                    batch_size=64, eps_dec=3e-6, replace=100, path_eval='./dueling_ddqn/slimevolley_dueling_ddqn_q_eval_no_bounce_5', greedy=True)
num_games = 300

list_rewards = np.zeros(num_games)
list_timesteps = np.zeros(num_games)
policy = slimevolleygym.BaselinePolicy()

for i in range(num_games):
    done = False
    total_reward = 0
    timestep = 0
    obs = env.reset()

    while not done:
        timestep += 1
        action1 = agent_right.choose_action(obs)
        action1 = action_table[action1]

        obs, reward, done, info = env.step(action1)

        total_reward += reward
        # env.render()

    list_rewards[i] = total_reward
    list_timesteps[i] = timestep
    print("Episodio:    ", i, "Score:   ", total_reward, "Timesteps:  ", timestep)


print("score:   ", np.mean(list_rewards))
print("timestep:    ", np.mean(list_timesteps))