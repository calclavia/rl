import gym
from models import *
from dqn import *
from pg import *
from a2c import *
from critic import *
from async import *
from optparse import OptionParser
from experience import *

import relay_generator

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
parser.add_option("-a", "--agent",  help="Agent")
parser.add_option("-r", "--run",  help="Run mode (no training)")

(options, args) = parser.parse_args()

env = gym.make(options.env)
# env.monitor.start('/tmp/cartpole-experiment-1')
learn = False if options.run else True

time_steps = 1
num_hidden = 100
layers = 10

# Create an agent based on the environment space.
agent = globals()[options.agent](
    env.observation_space,
    env.action_space,
    TemporalExperience(env.observation_space,  env.action_space, time_steps) if time_steps > 1 else Experience(env.observation_space,  env.action_space)
)

if time_steps > 1:
    agent.compile(rnn(
        space_to_shape(env.observation_space),
        time_steps,
        num_hidden,
        layers
    ))
else:
    agent.compile(dense(
        space_to_shape(env.observation_space),
        num_hidden,
        layers
    ))

agent.run(env, 10000, render=False, learn=learn)

# env.monitor.close()
