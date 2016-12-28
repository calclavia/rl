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
    inputs1, x1 = build_rnn(input_shape, num_outputs, time_steps, num_h)
    inputs2, x2 = build_rnn(input_shape, num_outputs, time_steps, num_h)
    # TODO: Combining network for more efficiency?
    policy_outputs = Dense(num_outputs, activation='softmax', name='output')(x1)
    value_output = Dense(1, activation='linear')(x2)

    advantages = Input(shape=(None,), name='advantages')

    model = Model(inputs1, policy_outputs)
    # Policy model
    policy_model = Model([inputs1, advantages], policy_outputs)
    policy_model.compile(RMSprop(), policy_loss(advantages))

    # Value model
    value_model = Model(inputs2, value_output)
    value_model.compile(RMSprop(), 'mse')

    print(model.summary())
    return model, policy_model, value_model

def preprocess(observation, ob_space):
    if isinstance(ob_space, spaces.Discrete):
        return one_hot(observation, ob_space.n)
    return observation

class DiscreteA2CAgent(Agent):

    def __init__(self, ob_space, action_space, discount=0.9, time_steps=1):
        super().__init__(ob_space, action_space)
        self.discount = discount
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

        self.values = deque(maxlen=100)
        self.advantages = deque(maxlen=100)

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
        observation = preprocess(observation, self.ob_space)
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

        # TODO: Implement tmax case
        if terminal:
            # last_state = list(self.temporal_memory)[1:] + [preprocess(observation, self.ob_space)]
            # Create training data
            discounted_rewards = discount_rewards(self.rewards, self.discount)
            current_states = np.array(self.observations)
            # next_states = np.array(self.dest_observations)
            # terminals = np.array(self.terminals)
            # next_values = self.value.predict(next_states).T[0]
            # target_values = rewards + self.discount * terminals * next_values

            #print(rewards, current_states, next_values, target_values)
            # Learn critic from TD-error
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
            self.dest_observations = []
            self.actions = []
            self.rewards = []
            self.terminals = []

        if terminal:
            if len(self.values) > 0:
                print('Value: {}, Advantage: {}'.format(
                    np.mean(self.values), np.mean(np.abs(self.advantages))))

            # TODO: Printing out all possible states. Remove this
            """
            all_states = [[one_hot(i, 16)] for i in range(16)]
            print('Value Table')
            values = self.value.predict(all_states).reshape(4, 4)
            print(values)

            print('Policy Table')
            policies = self.model.predict(all_states)
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
