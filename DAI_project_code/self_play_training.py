import gym
import slimevolleygym
from dqn import Agent
from utils import plotLearning
import matplotlib.pyplot as plt
import numpy as np

action_table = [[0, 0, 0],  # NOOP
                [1, 0, 0],  # LEFT (forward)
                [1, 0, 1],  # UPLEFT (forward jump)
                [0, 0, 1],  # UP (jump)
                [0, 1, 1],  # UPRIGHT (backward jump)
                [0, 1, 0]]  # RIGHT (backward)

if __name__ == '__main__':
    env = gym.make("SlimeVolley-v0")
    # agent_right = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12], lr=0.001)
    # agent_left = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12], lr=0.001)

    agent_right = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12], lr=0.001, path="./weights/weights_right_comp_512x512_3.pt")
    agent_left = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12], lr=0.001, path="./weights/weights_right_comp_512x512_3.pt")
    scores_right, eps_history_right = [], []
    scores_left, eps_history_left = [], []
    n_games = 6500
    time_steps_history = []

    for i in range(n_games):
        t = 0
        score_right = 0
        done = False
        observation_right = env.reset()
        observation_left = observation_right
        while not done:
            action_right = agent_right.choose_action(observation_right)
            action_left = agent_left.choose_action(observation_left)
            trio_action_right = action_table[action_right]
            trio_action_left = action_table[action_left]
            observation_, reward, done, info = env.step(trio_action_right, trio_action_left)
            observation_next_right = observation_
            observation_next_left = info['otherObs']
            score_right += reward
            agent_right.store_transition(observation_right, action_right, reward, observation_next_right, done)
            agent_right.learn()
            observation_right = observation_next_right
            observation_left = observation_next_left
            t += 1
            # env.render()
        scores_right.append(score_right)
        eps_history_right.append(agent_right.epsilon)
        time_steps_history.append(t)

        avg_score_right = np.mean(scores_right[-100:])

        if avg_score_right > 10:
            print("Updating left agent weights....")
            agent_left.Q_eval.load_state_dict(agent_right.Q_eval.state_dict())
        # if n_games % 50 == 0:
        #     print("Updating left agent weights....")
        #     agent_left.Q_eval.load_state_dict(agent_right.Q_eval.state_dict())

        print('RIGHT:       episode ', i, 'score %.2f' % score_right, 'average score %.2f' % avg_score_right, 'epsilon %.2f' % agent_right.epsilon)
        print('AVERAGE TIME STEPS:      %.2f' % np.mean(time_steps_history[-100:]))

        if n_games % 500 == 0:
            agent_right.save('./weights/weights_right_comp_512x512_4.pt')


    plt.scatter(range(len(time_steps_history)), time_steps_history)
    plt.show()

    x = [i+1 for i in range(n_games)]
    filename = './graphics/slimevolley_dqn_right_512x512_4.png'
    plotLearning(x, scores_right, eps_history_right, filename)
    str_time_steps = str(time_steps_history)
    print(str_time_steps)
    agent_right.save('./weights/weights_right_comp_512x512_4.pt')

    agent_left.Q_eval.load_state_dict(agent_right.Q_eval.state_dict())
    print(agent_right.Q_eval.state_dict())
    print(agent_left.Q_eval.state_dict())
