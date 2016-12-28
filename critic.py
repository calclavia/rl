import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import RMSprop
from collections import deque
from agent import Agent
from util import *


class CriticAgent(Agent):
    """
    Agent that learns the advantage function
    """

    def __init__(self,
                 ob_space,
                 action_space,
                 exp,
                 discount=0.9,
                 start_epsilon=1,
                 end_epsilon=0.1,
                 anneal_steps=10000):
        super().__init__(ob_space, action_space, exp, discount)
        # Epsilon
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.annealing = (start_epsilon - end_epsilon) / anneal_steps

    def compile(self, model):
        num_outputs = action_to_shape(self.action_space)

        inputs = model.input
        x = model.output
        output = Dense(1, activation='linear')(x)

        self.model = Model(inputs, output)
        self.model.compile(RMSprop(clipvalue=1.), 'mse')

    def choose(self):
        """
        The agent observes a state and chooses an action by the
        epsilon greedy policy.
        """
        # epsilon greedy exploration-exploitation
        if np.random.random() < self.epsilon:
            # Take a random action
            action = np.random.randint(self.action_space.n)
        else:
            state = self.exp.get_state()
            probabilities = self.model.predict(np.array([state]))[0]
            # Get q values for all actions in current state
            # Take the greedy policy (choose action with largest q value)
            action = np.argmax(probabilities)

        # Epsilon annealing
        if self.epsilon > self.end_epsilon:
            self.epsilon -= self.annealing

        return action

    def learn(self, terminal):
        # TODO: Implement tmax case, custom batch size?
        if terminal:
            # Learn critic
            rewards = discount_rewards(self.exp.rewards, self.discount)
            states = np.array(self.exp.get_states())

            self.model.fit(
                states,
                rewards,
                nb_epoch=1,
                verbose=0
            )
