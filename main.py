import gym
from dqn import *
from pg import *
from a3c import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
parser.add_option("-a", "--agent",  help="Agent")

(options, args) = parser.parse_args()

env = gym.make(options.env)
#env.monitor.start('/tmp/cartpole-experiment-1')

# Create an agent based on the environment space.
agent = globals()[options.agent](env.observation_space, env.action_space)
agent.fit(env, 10000)

#env.monitor.close()
