""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque
from agent import Agent
from util import *


def build_model(input_shape, num_outputs, time_steps, num_h=30):
    # Build Network
    inputs, x = build_rnn(input_shape, num_outputs, time_steps, num_h)
    outputs = Dense(num_outputs, activation='softmax', name='output')(x)

    advantages = Input(shape=(None,), name='advantages')

    model = Model(inputs, outputs)

    train_model = Model([inputs, advantages], outputs)
    train_model.compile(RMSprop(clipvalue=1), policy_loss(advantages))
    print(model.summary())
    return model, train_model


class PGAgent(Agent):

    def __init__(self, ob_space, action_space, discount=0.99, batch_size=32, time_steps=10):
        super().__init__(ob_space, action_space)
        self.discount = discount
        self.batch_size = batch_size
        self.time_steps = time_steps

        self.model, self.train_model = build_model(
            space_to_shape(ob_space),
            action_to_shape(action_space),
            self.time_steps
        )

        # Observations made
        self.observations = []
        # Actions taken
        self.actions = []
        # Rewards received
        self.rewards = []

    def run_episode(self, env, render=False, learn=True):
        # Fill in temporal memory
        self.temporal_memory = deque(maxlen=self.time_steps)
        for _ in range(self.time_steps - 1):
            self.temporal_memory.append(
                np.zeros(space_to_shape(self.ob_space)))

        super().run_episode(env, render and self.num_ep % self.batch_size == 0, learn)

    def forward(self, observation):
        if isinstance(self.ob_space, spaces.Discrete):
            observation = one_hot(observation, self.ob_space.n)

        # forward the policy network and sample an action from the returned
        # probability
        self.temporal_memory.append(observation)

        x = list(self.temporal_memory)
        prob_dist = self.model.predict(np.array([x]))[0]
        action = np.random.choice(prob_dist.size, p=prob_dist)

        # record various intermediates
        self.observations.append(x)
        self.actions.append(one_hot(action, self.action_space.n))
        return action

    def backward(self, observation, reward, terminal):
        # record reward
        self.rewards.append(reward)

        if terminal:
            if self.num_ep > 0 and self.num_ep % self.batch_size == 0:
                # compute the discounted reward backwards through time
                discounted_rewards = discount_rewards(
                    self.rewards, self.discount
                )

                advantages = z_score(discounted_rewards)
                targets = np.array(self.actions)

                self.train_model.fit(
                    [np.array(self.observations), advantages],
                    targets,
                    nb_epoch=1,
                    verbose=0
                )

                self.observations = []
                self.actions = []
                self.rewards = []
