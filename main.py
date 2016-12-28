import gym
from models import *
from dqn import *
from pg import *
from a2c import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
parser.add_option("-a", "--agent",  help="Agent")
parser.add_option("-r", "--run",  help="Run mode (no training)")

(options, args) = parser.parse_args()

env = gym.make(options.env)
# env.monitor.start('/tmp/cartpole-experiment-1')

learn = False if options.run else True

# Create an agent based on the environment space.
agent = globals()[options.agent](env.observation_space, env.action_space)

agent.compile(simple_rnn(
    space_to_shape(env.observation_space),
    5,
    20
))

agent.run(env, 10000, render=False, learn=learn)

# env.monitor.close()
