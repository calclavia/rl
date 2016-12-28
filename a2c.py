import numpy as np
from collections import deque
from agent import Agent
from critic import CriticAgent
from pg import PGAgent
from util import *


class A2CAgent(Agent):
    """
    Advantage actor critic agent that composes the policy gradient agent
    and the critic agent
    """

    def __init__(self, ob_space, action_space, time_steps=5):
        super().__init__(ob_space, action_space)
        self.time_steps = time_steps

        self.actor = PGAgent(ob_space, action_space, time_steps)
        self.critic = CriticAgent(ob_space, action_space, time_steps)

        # Replace compute_advantage function
        adv = self.actor.compute_advantage
        self.actor.compute_advantage = \
            lambda: adv() - self.critic.model.predict(self.actor.observations).T[0]

    def compile(self, model):
        self.critic.compile(model)
        self.actor.compile(model)

    def forward(self, observation):
        # TODO: We don't need this if we share experiences
        self.critic.forward(observation)
        return self.actor.forward(observation)

    def backward(self, observation, reward, terminal):
        self.critic.backward(observation, reward, terminal)
        self.actor.backward(observation, reward, terminal)

        # TODO: Printing out all possible states. Remove this
        """
        all_states = [[one_hot(i, 16)] for i in range(16)]
        print('Value Table')
        values = self.value.predict(all_states).reshape(4, 4)
        print(values)

        print('Policy Table')
        policies = self.predictor.predict(all_states)
        policies = np.argmax(policies, axis=1)
        print(policies.reshape(4, 4))

        print('Greedy Policy')
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        g_policy = np.zeros_like(values)
        for x in range(4):
            for y in range(4):
                # Find max value
                max_dir = -1
                max_val = float('-inf')
                for i, (dy, dx) in enumerate(dirs):
                    nX = x + dx
                    nY = y + dy
                    if nX >= 0 and nX < 4 and nY >= 0 and nY < 4:
                        v = values[nX, nY]
                        if v > max_val:
                            max_dir = i
                            max_val = v
                g_policy[x, y] = max_dir
        print(g_policy)
        """
