import gym
import slimevolleygym
from dqn import Agent
from utils import plotLearning
import numpy as np

action_table = [[0, 0, 0],  # NOOP
                [1, 0, 0],  # LEFT (forward)
                [1, 0, 1],  # UPLEFT (forward jump)
                [0, 0, 1],  # UP (jump)
                [0, 1, 1],  # UPRIGHT (backward jump)
                [0, 1, 0]]  # RIGHT (backward)

if __name__ == '__main__':
    env = gym.make("SlimeVolley-v0")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12], lr=0.001)
    scores, eps_history = [], []
    n_games = 1500
    policy = slimevolleygym.BaselinePolicy()

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            #action1 = policy.predict(observation)
            trio_action = action_table[action]
            #trio_action1 = action_table[action1]
            #observation_, reward, done, info = env.step(trio_action, action1)
            observation_, reward, done, info = env.step(trio_action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            #env.render()
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    filename = 'slimevolley_dqn.png'
    plotLearning(x, scores, eps_history, filename)
    agent.save('weights.pt')
