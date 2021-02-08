import gym
import slimevolleygym
from dqn import Agent
from utils import plotLearning
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent_right = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001)
    # agent_left = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12], lr=0.001)

    scores_right, eps_history_right = [], []
    n_games = 500

    for i in range(n_games):
        score_right = 0
        done = False
        observation_right = env.reset()
        while not done:
            action_right = agent_right.choose_action(observation_right)
            observation_, reward, done, info = env.step(action_right)
            observation_next_right = observation_
            score_right += reward
            agent_right.store_transition(observation_right, action_right, reward, observation_next_right, done)
            agent_right.learn()
            observation_right = observation_next_right
            # env.render()
        scores_right.append(score_right)
        eps_history_right.append(agent_right.epsilon)

        avg_score_right = np.mean(scores_right[-100:])

        print('RIGHT:       episode ', i, 'score %.2f' % score_right, 'average score %.2f' % avg_score_right, 'epsilon %.2f' % agent_right.epsilon)


    x = [i+1 for i in range(n_games)]
    filename = './graphics/lunar_lander_256x128.png'
    plotLearning(x, scores_right, eps_history_right, filename)
    agent_right.save('./weights/weights_lunar_lander_256x128.pt')

