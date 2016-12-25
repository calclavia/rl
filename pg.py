""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
from collections import deque
from agent import Agent
from util import *


class PGAgent(Agent):

    def __init__(self, ob_space, action_space, discount=0.9, batch_size=32, time_steps=5):
        super().__init__(ob_space, action_space)
        self.discount = discount
        self.batch_size = batch_size
        self.time_steps = time_steps

        self.model = build_rnn(ob_space.shape, self.time_steps)

        # Training buffer
        self.input_buffer, self.target_buffer = [], []

    def run_episode(self, env, mean_reward, render=False, learn=True):
        # reset array memory
        self.targets, self.rewards = [], []

        # Fill in temporal memory
        self.temporal_memory = deque(maxlen=self.time_steps)
        for _ in range(self.time_steps - 1):
            self.temporal_memory.append(np.zeros(self.ob_space.shape))

        return super().run_episode(env, mean_reward, render and self.num_ep % self.batch_size == 0, learn)

    def forward(self, observation):
        # forward the policy network and sample an action from the returned
        # probability
        self.temporal_memory.append(observation)

        x = list(self.temporal_memory)
        prob_dist = self.model.predict(np.array([x]))[0]
        action = np.random.choice(prob_dist.size, p=prob_dist)

        # record various intermediates
        self.input_buffer.append(x)

        # grad that encourages the action that was taken to be taken (see
        # http://cs231n.github.io/neural-networks-2/#losses if confused)
        # a "fake label"
        target = [1. if action == i else 0. for i in range(len(prob_dist))]

        self.targets.append(target)
        return action

    def backward(self, reward, terminal):
        # record reward
        self.rewards.append(reward)

        if terminal:
            # all inputs, action gradients, and rewards
            targets = np.vstack(self.targets)
            rewards = np.vstack(self.rewards)

            # compute the discounted reward backwards through time
            discounted_rewards = z_score(discount_rewards(rewards, self.discount))

            # modulate the gradient with advantage (PG magic happens right here.)
            # TODO: Is modulating the targe equiv? Maybe need to adjust loss
            # func
            targets *= discounted_rewards

            # Buffer
            self.target_buffer += targets.tolist()

            if self.num_ep % self.batch_size == 0:
                input_buffer = np.array(self.input_buffer)
                target_buffer = np.array(self.target_buffer)

                # TODO: Seems like epochs actually improve training speed?
                self.model.fit(input_buffer, target_buffer, verbose=0, nb_epoch=1)

                self.input_buffer = []
                self.target_buffer = []
