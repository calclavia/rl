import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.optimizers import RMSprop
from keras import backend as K


class DiscreteA3CAgent:

    def __init__(self, ob_space, action_space, discount=0.9, batch_size=32):
        super().__init__(ob_space, action_space)
        self.discount = discount

        self.policy, self.value = build_network(ob_space.shape, action_space.n)

        # Initialize experience
        self.states = []
        self.policy_distributions = []
        self.rewards = []

    def forward(self, observation):
        """
        Choose an action according to the policy
        """
        self.states.append(observation)
        probabilities = self.policy.predict(np.array([observation]))[0]
        self.policy_distributions.append(probabilities)
        # Sample action probabilities
        return np.random.choice(self.num_actions, p=probabilities)

    def backward(self, reward, terminal=False):
        # TODO: Clip reward?
        # Record reward
        self.rewards.append(reward)

        if terminal:
            # TOOD: Implement t max?
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

def build_network(input_shape, num_actions):
    # TODO: Consider LSTMs
    inputs = Input(shape=input_shape)
    #x = Flatten()(inputs)
    x = Dense(20, activation='relu')(inputs)

    # Policy prediction
    policy_output = Dense(num_actions, activation='softmax')(x)
    # Value function
    value_output = Dense(1, activation='linear')(x)

    policy_model = Model(inputs, policy_output)
    policy_model.compile(
        loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4, clipvalue=1)
    )

    value_model = Model(inputs, value_output)
    value_model.compile(
        loss='mse', optimizer=RMSprop(lr=1e-4, clipvalue=1)
    )
    return policy_model, value_model
