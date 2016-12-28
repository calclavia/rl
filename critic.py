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
                 start_epsilon=1,
                 end_epsilon=0.1,
                 anneal_steps=1000000,
                 time_steps=5):
        super().__init__(ob_space, action_space)
        self.time_steps = time_steps

        # Epsilon
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.annealing = (start_epsilon - end_epsilon) / anneal_steps

        # Observations made
        self.observations = []
        # Rewards received
        self.rewards = []

        # Fill in temporal memory
        self.temporal_memory = deque(maxlen=self.time_steps)
        for _ in range(self.time_steps - 1):
            self.temporal_memory.append(
                np.zeros(space_to_shape(self.ob_space)))

    def compile(self, model):
        num_outputs = action_to_shape(self.action_space)

        inputs = model.input
        x = model.output
        output = Dense(1, activation='linear')(x)

        self.model = Model(inputs, output)
        self.model.compile(RMSprop(), 'mse')

    def epsilon_greedy(self, probabilities):
        # epsilon greedy exploration-exploitation
        if np.random.random() < self.epsilon:
            # Take a random action
            action = np.random.randint(self.action_space.n)
        else:
            # Get q values for all actions in current state
            # Take the greedy policy (choose action with largest q value)
            action = np.argmax(probabilities)

        # Epsilon annealing
        if self.epsilon > self.end_epsilon:
            self.epsilon -= self.annealing

        return action

    def forward(self, observation):
        """
        The agent observes a state and chooses an action by the
        epsilon greedy policy.
        """
        # TODO: Abstract the temporal memory to agent
        observation = preprocess(observation, self.ob_space)
        self.temporal_memory.append(observation)

        state = list(self.temporal_memory)
        self.observations.append(state)
        predictions = self.model.predict(np.array([state]))[0]
        return self.epsilon_greedy(predictions)

    def backward(self, observation, reward, terminal):
        # record reward
        self.rewards.append(reward)

        # TODO: Implement tmax case, custom batch size?
        if terminal:
            # Learn critic
            discounted_rewards = discount_rewards(self.rewards, self.discount)
            states = np.array(self.observations)

            self.model.fit(
                states,
                discounted_rewards,
                nb_epoch=1,
                verbose=0
            )

            # Clear data
            self.observations = []
            self.rewards = []
