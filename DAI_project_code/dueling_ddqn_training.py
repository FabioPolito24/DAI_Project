import gym
import slimevolleygym
from dueling_ddqn import Agent
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
    num_games = 2000
    load_checkpoint = False

    # agent_right = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
    #               input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
    #               batch_size=64, eps_dec=2e-6, replace=100)
    # agent_left = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
    #               input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
    #               batch_size=64, eps_dec=3e-6, replace=100)

    agent_right = Agent(gamma=0.99, epsilon=0.0, lr=2e-5,
                  input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=5e-4, replace=100, path_eval='./dueling_ddqn/slimevolley_dueling_ddqn_q_eval_no_bounce_5',
                        path_next='./dueling_ddqn/slimevolley_dueling_ddqn_q_next_no_bounce_5')
    agent_left = Agent(gamma=0.99, epsilon=0.0, lr=2e-5,
                  input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=5e-4, replace=100, path_eval='./dueling_ddqn/slimevolley_dueling_ddqn_q_eval_no_bounce_5',
                        path_next='./dueling_ddqn/slimevolley_dueling_ddqn_q_next_no_bounce_5')

    filename = 'slimevolley-Dueling-DDQN-512-Adam-lr0005-replace100_5.png'
    scores, eps_history = [], []
    time_steps_history = []
    update_check = np.zeros(100, np.float32)
    update_counter = 0

    if load_checkpoint:
        agent_right.load_models()
        agent_left.load_models()

    for i in range(num_games):
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
            agent_right.store_transition(observation_right, action_right, reward, observation_next_right, int(done))
            agent_right.learn()
            observation_right = observation_next_right
            observation_left = observation_next_left
            t += 1
            # env.render()
        scores.append(score_right)
        eps_history.append(agent_right.epsilon)
        time_steps_history.append(t)

        avg_score = np.mean(scores[-100:])

        update_counter += 1
        update_check[update_counter % 100] = score_right

        if update_counter > 100 and np.mean(update_check) > 10:
            update_counter = 0
            print("Updating left agent weights....")
            agent_left.q_eval.load_state_dict(agent_right.q_eval.state_dict())
            # agent_left.load_models()

        # if n_games % 50 == 0:
        #     print("Updating left agent weights....")
        #     agent_left.Q_eval.load_state_dict(agent_right.Q_eval.state_dict())

        print('RIGHT:       episode ', i, 'score %.2f' % score_right, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent_right.epsilon)
        print('AVERAGE TIME STEPS:      %.2f' % np.mean(time_steps_history[-100:]))

        if i % 500 == 0:
            print('Saving weights....')
            #agent_right.save_models('./weights/weights_comp_duelling_512.pt')
            agent_right.save_models()

    plt.scatter(range(len(time_steps_history)), time_steps_history)
    plt.show()

    x = [i+1 for i in range(num_games)]
    filename = './graphics/slimevolley_comp_duelling_512_6.png'
    plotLearning(x, scores, eps_history, filename)
    str_time_steps = str(time_steps_history)
    print(str_time_steps)
    agent_right.save_models()

    agent_left.q_eval.load_state_dict(agent_right.q_eval.state_dict())
    print(agent_right.q_eval.state_dict())
    print(agent_left.q_eval.state_dict())
