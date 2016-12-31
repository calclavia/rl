import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import os
import gym

from keras import backend as K
from keras.layers import Dense
from models import *
from util import *

from optparse import OptionParser

import relay_generator

class AC_Network():
    def __init__(self, state_space, a_size, scope, optimizer):
        with tf.variable_scope(scope):
            self.inputs, x = dense(state_space, 50, 1)

            #Output layers for policy and value estimations
            self.policy = Dense(a_size, activation='softmax', name='policy_output')(x)
            self.value = Dense(1, activation='linear', name='value_output')(x)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))

class A3CAgent:
    def __init__(self, env, name, model, discount, callbacks):
        # Name of the worker
        self.name = name
        # Discount factor
        self.discount = discount
        # Graph the metrics
        self.metrics = {
            'value_loss': [],
            'policy_loss': [],
            'grad_norm': [],
            'var_norm': [],
            'rewards': [],
            'lengths': [],
            'mean_values': []
        }

        self.callbacks = callbacks

        # Local copy of the model
        self.model = model
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = env

    def train(self,
            sess,
            observations,
            actions,
            rewards,
            next_observations,
            values,
            bootstrap_value):

        # Here we take the rewards and values from the exp, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = rewards + [bootstrap_value]
        discounted_rewards = discount(rewards_plus, self.discount)[:-1]
        value_plus = np.array(values + [bootstrap_value])
        advantages = rewards + self.discount * value_plus[1:] - value_plus[:-1]
        advantages = discount(advantages, self.discount)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                self.model.value_loss,
                self.model.policy_loss,
                self.model.entropy,
                self.model.grad_norms,
                self.model.var_norms,
                self.model.apply_grads
            ],
            {
                self.model.target_v: discounted_rewards,
                self.model.inputs: np.vstack(observations),
                self.model.actions: actions,
                self.model.advantages: advantages,
                K.learning_phase(): 1
            }
        )

        N = len(observations)

        # Record metrics
        self.metrics['value_loss'].append(v_l / N)
        self.metrics['policy_loss'].append(p_l / N)
        self.metrics['grad_norm'].append(g_n / N)
        self.metrics['var_norm'].append(v_n / N)

    def work(self, max_episode_length, sess, coord):
        self.episode_count = 0
        total_steps = 0
        print("Starting worker " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # Sync local network with global network
                sess.run(self.update_local_ops)

                # Buffer the data obtained during the episode
                values = []
                states = []
                next_states = []
                actions = []
                rewards = []

                total_reward = 0
                total_value = 0
                episode_step_count = 0
                done = False

                observation = self.env.reset()

                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v = sess.run([
                            self.model.policy,
                            self.model.value
                        ],
                        {
                            self.model.inputs: [observation],
                            K.learning_phase(): 0
                        }
                    )

                    action = np.random.choice(a_dist[0], p=a_dist[0])
                    action = np.argmax(a_dist == action)
                    value = v[0, 0]

                    next_observation, reward, done, info = self.env.step(action)

                    # Bookkeeping
                    states.append(observation)
                    next_states.append(next_observation)
                    rewards.append(reward)
                    actions.append(action)
                    values.append(value)

                    total_value += value
                    total_reward += reward
                    observation = next_observation
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is
                    # full, then we make an update step using that experience.
                    if len(states) == 30 and not done and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is,
                        # we "bootstrap" from our current value estimation.
                        v1 = sess.run(
                            self.model.value,
                            {
                                self.model.inputs:[observation],
                                K.learning_phase(): 0
                            }
                        )[0,0]

                        self.train(
                            sess,
                            states,
                            actions,
                            rewards,
                            next_states,
                            values,
                            v1
                        )

                        values = []
                        states = []
                        next_states = []
                        actions = []
                        rewards = []

                        sess.run(self.update_local_ops)

                # Train the network using the experience buffer at the end of the episode.
                self.train(
                    sess,
                    states,
                    actions,
                    rewards,
                    next_states,
                    values,
                    0.0
                )

                # Record metrics
                self.metrics['rewards'].append(total_reward)
                self.metrics['lengths'].append(episode_step_count)
                self.metrics['mean_values'].append(total_value / episode_step_count)

                # Execute callbacks
                for cb in self.callbacks:
                    cb(self)

                self.episode_count += 1

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = False
output_path = './out'
model_path = output_path + '/model'

# Set workers ot number of available CPU threads
num_workers = multiprocessing.cpu_count()

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
(options, args) = parser.parse_args()

env = gym.make(options.env)
# Observation space size
state_space = space_to_shape(env.observation_space)
# Agent action space size
a_size = action_to_shape(env.action_space)

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"),  tf.Session() as sess:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    # Generate global network
    master_network = AC_Network(state_space, a_size, 'global', None)
    saver = tf.train.Saver(max_to_keep=5)
    workers = []

    def saver_fn(agent):
        if agent.episode_count % 100 == 0:
            saver.save(sess, model_path + '/model-' + str(agent.episode_count) + '.cptk')

    # Create worker classes
    for i in range(num_workers):
        name = 'worker_' + str(i)
        model = AC_Network(state_space, a_size, name, optimizer)
        writer_fn = summary_writer('', tf.summary.FileWriter(output_path + '/tb/' + name))

        if i == 0:
            cbs = [writer_fn, saver_fn]
        else:
            cbs = [writer_fn]

        workers.append(A3CAgent(gym.make(options.env), name, model, gamma, cbs))

    coord = tf.train.Coordinator()

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, sess, coord)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
