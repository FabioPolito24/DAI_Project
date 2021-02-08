import gym
import slimevolleygym
from dueling_ddqn import Agent
from gym.wrappers.monitoring.video_recorder import VideoRecorder

action_table = [[0, 0, 0],  # NOOP
                [1, 0, 0],  # LEFT (forward)
                [1, 0, 1],  # UPLEFT (forward jump)
                [0, 0, 1],  # UP (jump)
                [0, 1, 1],  # UPRIGHT (backward jump)
                [0, 1, 0]]  # RIGHT (backward)

env = gym.make("SlimeVolley-v0")


agent_right = Agent(gamma=0.99, epsilon=0, lr=5e-4,
                    input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
                    batch_size=64, eps_dec=3e-6, replace=100, path_eval='./dueling_ddqn/slimevolley_dueling_ddqn_q_eval_no_bounce_9', greedy=True)
agent_left = Agent(gamma=0.99, epsilon=0, lr=5e-4,
                   input_dims=[12], n_actions=6, mem_size=100000, eps_min=0.01,
                   batch_size=64, eps_dec=3e-6, replace=100, path_eval='./dueling_ddqn/slimevolley_dueling_ddqn_q_eval_no_bounce_9', greedy=True)

obs1 = env.reset()
obs2 = obs1 # both sides always see the same initial observation.

video_recorder = None
video_recorder = VideoRecorder(env, "./video/comp_15.mp4", enabled=True)

done = False
total_reward = 0
policy = slimevolleygym.BaselinePolicy()

while not done:

  action1 = agent_right.choose_action(obs1)
  #action2 = agent_left.choose_action(obs2)
  action1 = action_table[action1]
  #action2 = action_table[action2]
  action2 = policy.predict(obs2)

  obs1, reward, done, info = env.step(action1, action2) # extra argument
  obs2 = info['otherObs']

  total_reward += reward
  env.render()
  video_recorder.capture_frame()

print("policy1's score:", total_reward)
print("policy2's score:", -total_reward)

video_recorder.close()
video_recorder.enabled = False
env.close()