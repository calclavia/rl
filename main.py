import gym
import numpy as np
import time
from dqn import Agent
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")

(options, args) = parser.parse_args()

env = gym.make(options.env)
#env.monitor.start('/tmp/cartpole-experiment-1')

# Create an agent based on the environment.
agent = Agent(
    state_shape=env.observation_space.shape,
    num_actions=env.action_space.n
)

num_episodes = 10000

for e in range(num_episodes):
    state = env.reset()
    done = False
    total_loss = 0
    total_reward = 0
    max_q = 0
    i = 0
    t = time.time()

    while not done:
        #env.render()
        # Choose an action
        action, prev_state, predictions = agent.choose(state)
        # Perform action
        state, reward, done, info = env.step(action)
        # Observe results of chosen action
        agent.observe(prev_state, action, reward, state, done)
        # Learn based on past experience
        total_loss += agent.learn()
        total_reward += reward
        max_q += np.max(predictions)
        i += 1

    max_q /= i

    print('Episode {}: Reward={} Loss={} Max Q={} Time={}'.format(e, total_reward, total_loss, max_q, time.time() - t))

#env.monitor.close()
