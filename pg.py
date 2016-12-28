import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import RMSprop
from keras import backend as K
from agent import Agent
from util import *

class PGAgent(Agent):
    """
    Policy gradient agent
    """
    def compile(self, model):
        num_outputs = action_to_shape(self.action_space)

        inputs = model.input
        x = model.output
        outputs = Dense(num_outputs, activation='softmax', name='output')(x)
        advantages = Input(shape=(None,), name='advantages')

        # Prediction model
        self.model = Model(inputs, outputs)

        # Training model
        def loss(target, output):
            # Target is a one-hot vector of actual action taken
            # Crossentropy weighted by advantage
            responsible_outputs = K.sum(output * target, 1)
            policy_loss = -K.sum(K.log(responsible_outputs) * advantages)
            entropy = -K.sum(output * K.log(output), 1)
            return policy_loss - 0.01 * entropy

        self.trainer = Model([inputs, advantages], outputs)
        self.trainer.compile(RMSprop(clipvalue=1.), loss)

    def choose(self):
        """
        Choose an action according to the policy
        """
        state = self.exp.get_state()
        prob_dist = self.model.predict(np.array([state]))[0]
        return np.random.choice(prob_dist.size, p=prob_dist)

    def learn(self, terminal):
        # TODO: Implement tmax case, custom batch size?
        if terminal:
            # Learn policy
            states = np.array(self.exp.get_states())
            advantages = self.compute_advantage()
            targets = np.array(self.exp.actions)

            self.trainer.fit(
                [states, advantages],
                targets,
                nb_epoch=1,
                verbose=0
            )

    def compute_advantage(self):
        return discount_rewards(self.exp.rewards, self.discount)
