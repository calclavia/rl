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
parser.add_option("-p", "--episodes",  help="Number of episodes")

parser.add_option("-t", "--time_steps",  help="Number of LSTM time steps")
parser.add_option("-n", "--hidden",  help="Number of hidden units")
parser.add_option("-l", "--layers",  help="Number of layers")

(options, args) = parser.parse_args()

env = gym.make(options.env)
# env.monitor.start('/tmp/cartpole-experiment-1')
learn = False if options.run else True

time_steps = int(options.time_steps) if options.time_steps else 1
num_hidden = int(options.hidden) if options.hidden else 100
layers = int(options.layers) if options.layers else 10
episodes = int(options.episodes) if options.episodes else 10000

# Create an agent based on the environment space.
agent = globals()[options.agent](
    env.observation_space,
    env.action_space,
    TemporalExperience(env.observation_space,  env.action_space, time_steps) if time_steps > 1 else Experience(
        env.observation_space,  env.action_space)
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

agent.run(env, episodes, render=False, learn=learn)

# env.monitor.close()
