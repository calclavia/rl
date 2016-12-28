import time
import numpy as np
from collections import deque
from gym import spaces

class Agent:
    def __init__(self, ob_space, action_space, exp, discount=0.9):
        self.ob_space = ob_space
        self.action_space = action_space
        self.discount = discount

        # The agent's past experience
        self.exp = exp

    def compile(self, model):
        """
        Setup the model
        """
        self.model = model

    def preprocess(self, observation):
        """
        Preprocesses the input observation before recording it into experience
        """
        if isinstance(self.ob_space, spaces.Discrete):
            return one_hot(observation, self.ob_space.n)
        return observation

    def run(self, env, num_episodes, render=False, learn=True):
        """
        Creates an environment for the agent to run in
        """
        self.total_rewards = deque(maxlen=100)

        for self.num_ep in range(num_episodes):
            self.run_episode(env, render, learn)

    def run_episode(self, env, render=False, learn=True):
        """
        Runs this agent in an environment
        """
        observation = env.reset()
        done = False
        total_reward = 0
        t = time.time()
        self.step = 0

        while not done:
            if render:
                env.render()
            observation = self.preprocess(observation)

            self.exp.observe(observation)

            # Choose an action
            action = self.choose()
            self.exp.act(action)

            # Perform action
            observation, reward, done, info = env.step(action)
            self.exp.reward(reward)

            if learn:
                # Observe results of chosen action
                self.learn(done)

            total_reward += reward
            self.step += 1

        self.total_rewards.append(total_reward)
        self.exp.clear()

        print('Episode {}: Reward={} ({}) Time={}'.format(
            self.num_ep,
            total_reward,
            np.mean(self.total_rewards),
            time.time() - t
        ))

    def choose(self):
        return self.action_space.sample()

    def learn(self, terminal):
        pass

    def load(self):
        """
        Loads the agent's model
        """
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print("Loading weights from {}.h5".format(self.save_name))
        except:
            print("Training a new model")

    def save(self):
        """
        Saves the agent's model
        """
        # TODO: Decouple save logic?
        self.model.save_weights('{}.h5'.format(self.save_name), True)
