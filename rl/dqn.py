"""
Implementation of Deep Q Network.
TODO: Incomplete WIP
"""
import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model

from .agent import Agent
from .memory import Memory

NUM_EPISODES = 100000  # Number of episodes the agent plays
GAMMA = 0.99  # Discount factor
# Number of steps to populate the replay memory before training starts
INITIAL_REPLAY_SIZE = 1000
NUM_REPLAY_MEMORY = 10000  # Number of replay memory the agent uses for training
BATCH_SIZE = 16  # Mini batch size
# The frequency with which the target network is updated
TARGET_UPDATE_INTERVAL = 100
TRAIN_INTERVAL = 1  # The agent selects 4 actions between successive updates
# Constant added to the squared gradient in the denominator of the RMSProp
# update
MIN_GRAD = 0.01
SAVE_SUMMARY_PATH = 'out/summary/'

# TODO: Depend only on Keras?
# TODO: Timestep support
# TODO: Multi-input support
class DQNAgent(Agent):

    def __init__(self,
                 model_builder,
                 initial_epsilon=1,
                 final_epsilon=0.1,
                 explore_steps=1000000,
                 preprocess=lambda x: x):
        """
        Args
            model_builder: A function that create a new model for the network
            initial_epsilon: Starting epsilon
            final_epsilon: Ending epsilon
            explore_steps: Number of steps over which the initial value of
                           epsilon is linearly annealed to its final value
            preprocess: Function called to preprocess observations
        """
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_step = (initial_epsilon - final_epsilon) / explore_steps
        self.explore_steps = explore_steps
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.memory = deque()

        # Misc
        self.model_builder = model_builder
        self.preprocess = preprocess

    def compile(self, sess, optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):
        self.sess = sess

        # Create q network
        self.q_model = self.model_builder()
        self.num_actions = self.q_model.outputs[0].get_shape()[1]
        q_weights = self.q_model.trainable_weights

        # Create target network
        self.t_model = self.model_builder()
        t_weights = self.t_model.trainable_weights

        # Syncs the target Q network's weight with the Q network's weights
        self.sync = [t_weights[i].assign(q_weights[i])
                     for i in range(len(t_weights))]

        # Define loss and gradient update operation
        self.a = tf.placeholder(tf.int64, [None])
        self.y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(self.a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(
            tf.mul(self.q_model.outputs[0], a_one_hot), reduction_indices=1
        )

        # Clip the error, the loss is quadratic when the error is in (-1, 1),
        # and linear outside of that region
        error = tf.abs(self.y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        # Define loss and gradient update operation
        self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        self.train_op = optimizer.minimize(self.loss, var_list=q_weights)

        # Setup metrics
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            SAVE_SUMMARY_PATH, self.sess.graph
        )

        # Init vars
        self.sess.run(tf.global_variables_initializer())

        # Initialize target network
        self.sess.run(self.sync)

    def train(self, sess, env_builder):
        env = env_builder()

        for _ in range(NUM_EPISODES):
            terminal = False
            state = self.preprocess(env.reset())
            while not terminal:
                action = self.get_action(state)
                next_state, reward, terminal, _ = env.step(action)
                next_state = self.preprocess(next_state)
                self.run(state, action, reward, terminal, next_state)
                state = next_state

    def get_action(self, state):
        """
        Picks an action given a state based on epsilon greedy policy.
        """
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_model.predict(np.array([state]))[0])

        # Anneal epsilon linearly over time
        if self.epsilon > self.final_epsilon and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step
        return action

    def run(self, state, action, reward, terminal, next_state):
        # Clip all positive rewards at 1 and all negative rewards at -1,
        # leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.memory.append((state, action, reward, next_state, terminal))

        if len(self.memory) > NUM_REPLAY_MEMORY:
            self.memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.learn()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.sync)

        self.total_reward += reward
        self.total_q_max += np.amax(self.q_model.predict(np.array([state]))[0])
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                         self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            """
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + self.explore_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))
            """

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def learn(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.t_model.predict(np.array(next_state_batch))
        y_batch = reward_batch + (1 - terminal_batch) * \
            GAMMA * np.amax(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.train_op], {
            self.q_model.inputs[0]: np.array(state_batch),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary('/Total Reward/Episode',
                          episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary(
            '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.scalar_summary('/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary('/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward,
                        episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(
            tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(
            summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
