import time
import numpy as np
from collections import deque

class Agent:
    def __init__(self, ob_space, action_space):
        self.ob_space = ob_space
        self.action_space = action_space

    def run(self, env, num_episodes, render=False, learn=True):
        """
        Fits this agent to the environment
        """
        self.total_rewards = deque(maxlen=100)

        for self.num_ep in range(num_episodes):
            self.run_episode(env, render, learn)

    def run_episode(self, env, render=False, learn=True):
        observation = env.reset()
        done = False
        total_reward = 0
        t = time.time()
        self.step = 0

        while not done:
            if render:
                env.render()
            # Choose an action
            action = self.forward(observation)
            # Perform action
            observation, reward, done, info = env.step(action)

            if learn:
                # Observe results of chosen action
                self.backward(observation, reward, done)
            total_reward += reward
            self.step += 1

        self.total_rewards.append(total_reward)

        print('Episode {}: Reward={} ({}) Time={}'.format(
            self.num_ep,
            total_reward,
            np.mean(self.total_rewards),
            time.time() - t
        ))

    def forward(self, observation):
        pass

    def backward(self, reward, terminal):
        pass
