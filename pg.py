import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque
from agent import Agent
from util import *


class PGAgent(Agent):
    """
    Policy gradient agent
    """

    def __init__(self, ob_space, action_space, time_steps=5):
        super().__init__(ob_space, action_space)
        self.time_steps = time_steps

        # Observations made
        self.observations = []
        # Actions taken
        self.actions = []
        # Rewards received
        self.rewards = []

    def compile(self, model):
        super().compile(model)
        num_outputs = action_to_shape(self.action_space)

        inputs = model.input
        x = model(inputs)
        policy_outputs = Dense(
            num_outputs, activation='softmax', name='output')(x)
        advantages = Input(shape=(None,), name='advantages')

        # Predicting model
        self.predictor = Model(inputs, policy_outputs)
        # Training model
        self.trainer = Model([inputs, advantages], policy_outputs)
        self.trainer.compile(RMSprop(), policy_loss(advantages))

    def run_episode(self, env, render, learn):
        # Fill in temporal memory
        self.temporal_memory = deque(maxlen=self.time_steps)
        for _ in range(self.time_steps - 1):
            self.temporal_memory.append(
                np.zeros(space_to_shape(self.ob_space)))

        super().run_episode(env, render, learn)

    def forward(self, observation):
        """
        Choose an action according to the policy
        """
        observation = preprocess(observation, self.ob_space)
        self.temporal_memory.append(observation)

        x = list(self.temporal_memory)
        prob_dist = self.predictor.predict(np.array([x]))[0]
        action = np.random.choice(prob_dist.size, p=prob_dist)

        # record data
        self.observations.append(x)
        self.actions.append(one_hot(action, self.action_space.n))
        return action

    def backward(self, observation, reward, terminal):
        # record reward
        self.rewards.append(reward)

        # TODO: Implement tmax case, custom batch size?
        if terminal:
            # Learn policy
            states = np.array(self.observations)
            advantages = self.compute_advantage()
            targets = np.array(self.actions)

            self.trainer.fit(
                [states, advantages],
                targets,
                nb_epoch=1,
                verbose=0
            )

            # Clear data
            self.observations = []
            self.actions = []
            self.rewards = []

    def compute_advantage(self):
        return discount_rewards(self.rewards, self.discount)
