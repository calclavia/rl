import gym
import numpy as np
import time
from a3c import DiscreteA3CAgent
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")

(options, args) = parser.parse_args()

def work():
    env = gym.make(options.env)
    #env.monitor.start('/tmp/cartpole-experiment-1')

    # Create an agent based on the environment.
    agent = DiscreteA3CAgent(env.observation_space.shape, env.action_space.n)

    num_episodes = 10000

    for e in range(num_episodes):
        state = env.reset()
        done = False
        total_loss = 0
        total_reward = 0
        t = time.time()

        while not done:
            env.render()
            # Choose an action
            action = agent.forward(state)
            # Perform action
            state, reward, done, info = env.step(action)
            # Observe results of chosen action
            agent.backward(reward, done)
            total_reward += reward

        print('Episode {}: Reward={} Loss={} Time={}'.format(e, total_reward, total_loss, time.time() - t))

    #env.monitor.close()

work()
