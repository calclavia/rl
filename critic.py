import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque
from agent import Agent
from util import *

def build_ac_model(input_shape, num_outputs, time_steps, num_h):
    # Build Network
    inputs, x = build_rnn(input_shape, num_outputs, time_steps, num_h)
    policy_outputs = Dense(num_outputs, activation='softmax', name='output')(x)
    value_output = Dense(1, activation='linear')(x)
    advantages = Input(shape=(None,), name='advantages')

    predictor = Model(inputs, policy_outputs)
    # Policy model
    policy_model = Model([inputs, advantages], policy_outputs)
    policy_model.compile(RMSprop(), policy_loss(advantages))

    # Value model
    value_model = Model(inputs, value_output)
    value_model.compile(RMSprop(), 'mse')

    print(predictor.summary())
    return predictor, policy_model, value_model

class DiscreteA2CAgent(Agent):

    def __init__(self, ob_space, action_space, discount=0.9, time_steps=5):
        super().__init__(ob_space, action_space)
        self.discount = discount
        self.time_steps = time_steps

        self.predictor, self.policy, self.value = build_ac_model(
            space_to_shape(ob_space),
            action_to_shape(action_space),
            time_steps,
            20
        )

        # Observations made
        self.observations = []
        # Actions taken
        self.actions = []
        # Rewards received
        self.rewards = []

        self.values = deque(maxlen=100)
        self.advantages = deque(maxlen=100)

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

        # record various intermediates
        self.observations.append(x)
        self.actions.append(one_hot(action, self.action_space.n))
        return action

    def backward(self, observation, reward, terminal):
        # record reward
        self.rewards.append(reward)

        # TODO: Implement tmax case, custom batch size?
        if terminal:
            # Learn critic
            # TODO: Duplicate
            discounted_rewards = discount_rewards(self.rewards, self.discount)
            current_states = np.array(self.observations)

            self.value.fit(
                current_states,
                discounted_rewards,
                nb_epoch=1,
                verbose=0
            )

            # Learn policy
            current_values = self.value.predict(current_states).T[0]
            advantages = discounted_rewards - current_values
            targets = np.array(self.actions)

            self.policy.fit(
                [current_states, advantages],
                targets,
                nb_epoch=1,
                verbose=0
            )

            self.values.append(np.mean(current_values))
            self.advantages.append(np.mean(np.abs(advantages)))

            # Clear data
            self.observations = []
            self.actions = []
            self.rewards = []

            print('Value: {}, Advantage: {}'.format(np.mean(self.values), np.mean(np.abs(self.advantages))))

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
