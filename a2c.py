import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque
from agent import Agent
from util import *

def build_model(input_shape, num_outputs, time_steps, num_h=30):
    # Build Network
    inputs = Input(shape=(time_steps,) + input_shape, name='input')
    x = LSTM(num_h, activation='relu', name='hidden')(inputs)
    policy_outputs = Dense(num_outputs, activation='softmax', name='output')(x)
    value_output = Dense(1, activation='linear')(x)

    advantages = Input(shape=(None,), name='advantages')

    model = Model(inputs, policy_outputs)
    # Policy model
    policy_model = Model([inputs, advantages], policy_outputs)

    policy_model.compile(RMSprop(1e-4), policy_loss(advantages))

    # Value model
    value_model = Model(inputs, value_output)
    value_model.compile(RMSprop(1e-4), 'mse')
    print(model.summary())
    return model, policy_model, value_model


class DiscreteA2CAgent(Agent):

    def __init__(self, ob_space, action_space, discount=0.99, batch_size=32, time_steps=5):
        super().__init__(ob_space, action_space)
        self.discount = discount
        self.batch_size = batch_size
        self.time_steps = time_steps

        self.model, self.policy, self.value = build_model(
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
        # Terminals
        self.terminals = []

        self.values = deque(maxlen=100)
        self.td_errors = deque(maxlen=100)

    def run_episode(self, env, render=False, learn=True):
        # Fill in temporal memory
        self.temporal_memory = deque(maxlen=self.time_steps)
        for _ in range(self.time_steps - 1):
            self.temporal_memory.append(
                np.zeros(space_to_shape(self.ob_space)))

        super().run_episode(env, render and self.num_ep % self.batch_size == 0, learn)

    def forward(self, observation):
        """
        Choose an action according to the policy
        """
        if isinstance(self.ob_space, spaces.Discrete):
            observation = one_hot(observation, self.ob_space.n)

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
        self.terminals.append(0 if terminal else 1)

        if self.step > 0 and self.step % self.batch_size == 0:
            rewards = np.array(self.rewards[:-1])
            current_states = np.array(self.observations[:-1])
            next_states = np.array(self.observations[1:])
            terminals = np.array(self.terminals[:-1])

            next_values = self.value.predict(next_states).T[0]
            target_values = rewards + self.discount * terminals * next_values

            # Learn critic from TD-error
            # TODO: Seems like TD error is decreasing correctly
            self.value.fit(
                current_states,
                target_values,
                nb_epoch=1,
                verbose=0
            )

            # Learn policy
            current_values = self.value.predict(current_states).T[0]
            td_error = target_values - current_values
            self.values.append(np.mean(current_values))
            self.td_errors.append(np.mean(td_error))

            # TODO: Decay
            # The advantage is the TD error
            advantages = z_score(td_error)
            targets = np.array(self.actions[:-1])

            self.policy.fit(
                [current_states, advantages],
                targets,
                nb_epoch=1,
                verbose=0
            )

        if terminal:
            print('Average value: {}, TD error: {}'.format(np.mean(self.values), np.mean(self.td_errors)))

        """
        R = 0
        grad_policy = 0
        grad_value = 0

        # Go from back to start
        exp = zip(self.states, self.policy_distributions, self.rewards)
        for i, (state, policy_dist, reward) in enumerate(reversed(exp)):
            R = reward + self.discount * R

            # Accumulate gradients
            grad_policy += np.log(policy_dist) * \
                (R - self.value.predict([state])[0])
            grad_value += 0  # TODO
        """
