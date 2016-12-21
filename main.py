import gym
import numpy as np
import time
from dqn import Agent

env = gym.make('CartPole-v0')
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
    total_reward = 0.0
    t = time.time()

    while not done:
        #env.render()
        # Choose an action
        action, prev_state = agent.choose(state)
        # Perform action
        state, reward, done, info = env.step(action)
        # Observe results of chosen action
        agent.observe(prev_state, action, reward, state, done)
        # Learn based on past experience
        agent.learn()
        total_reward += reward

    print('Episode {}: Reward={} Time={}'.format(e, total_reward, time.time() - t))

#env.monitor.close()
