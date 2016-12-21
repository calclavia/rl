import gym
import numpy as np
from dqn import Agent

env = gym.make('CartPole-v0')

# Create an agent based on the environment.
agent = Agent(
    state_shape=env.observation_space.shape,
    num_actions=env.action_space.n
)

num_episodes = 10000

for e in range(num_episodes):
    state = env.reset()
    done = False
    total_cost = 0.0
    total_reward = 0.0

    while not done:
        # TODO: Time episodes
        #env.render()
        # Choose an action
        action, prev_state = agent.choose(state)
        # Perform action
        state, reward, done, info = env.step(action)
        # Observe results of chosen action
        agent.observe(prev_state, action, reward, state)
        # Learn based on past experience
        agent.learn(done)
        total_reward += reward

    print('Episode {}: Reward={}'.format(e, total_reward))
