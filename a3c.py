import numpy as np
import tensorflow as tf
import os
import gym
import threading
import multiprocessing
import time
from gym import spaces

from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from .util import *
from .memory import Memory

class ACModel:
    def __init__(self, model_builder, scope, beta):
        self.scope = scope
        # Entropy weight
        self.beta = beta

        with tf.variable_scope(self.scope):
            self.model = model_builder()
            # Output layers for policy and value estimations
            self.policies = self.model.outputs[:-1]
            self.value = self.model.outputs[-1]

    def compile(self, optimizer, grad_clip):
        # Only the worker network need ops for loss functions and gradient updating.
        with tf.variable_scope(self.scope):
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            # Action chosen for every single policy output
            self.actions = []
            policy_losses = []
            entropies = []

            # Every policy output
            for policy in self.policies:
                num_actions = policy.get_shape()[1]
                action = tf.placeholder(shape=[None], dtype=tf.int32)
                actions_hot = tf.one_hot(action, num_actions)
                self.actions.append(action)

                responsible_outputs = tf.reduce_sum(policy * actions_hot, [1])
                # Entropy regularization
                # TODO: Clipping should be configurable
                entropies.append(-tf.reduce_sum(policy * tf.log(tf.clip_by_value(policy, 1e-20, 1.0))))
                # Policy loss
                policy_losses.append(-tf.reduce_sum(tf.log(responsible_outputs) * self.advantages))

            # Compute average policy and entropy loss
            self.policy_loss = tf.reduce_sum(policy_losses)
            self.entropy = tf.reduce_sum(entropies)

             # Value loss (Mean squared error)
            self.value_loss = tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            # Learning rate for Critic is half of Actor's, so multiply by 0.5
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.beta * self.entropy

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            # Clip norm of gradients
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, grad_clip)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.train = optimizer.apply_gradients(zip(grads, global_vars))

class A3CAgent:
    def __init__(self,
                 model_builder,
                 time_steps=0,
                 preprocess=lambda e, x: x,
                 model_path='out/model',
                 entropy_factor=1e-2,
                 batch_size=32):
        self.model_builder = model_builder
        self.time_steps = time_steps
        self.model_path = model_path
        self.entropy_factor = entropy_factor
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.save_count = 0
        # Generate global network
        self.model = ACModel(model_builder, 'global', entropy_factor)
        self.saver = tf.train.Saver(max_to_keep=5)

    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save(self, sess):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.saver.save(sess, self.model_path + '/model-' + str(self.save_count) + '.cptk')
        self.save_count += 1

    def train(self,
              env_name,
              grad_clip=50.,
              discount=.99,
              summary_path='out/summary/',
              num_workers=multiprocessing.cpu_count(),
              optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):
        print('Training model')

        self.model.compile(optimizer, grad_clip)
        print(self.model.model.summary())

        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # TODO: Not SRP. Move this.
            try:
                self.load(sess)
                print('Loading last saved session')
            except:
                print('Starting new session')

            coord = tf.train.Coordinator()
            worker_threads = []

            for i in range(num_workers):
                name = 'worker_' + str(i)
                writer = tf.summary.FileWriter(summary_path + name, sess.graph, flush_secs=2)
                t = threading.Thread(target=self.train_thread, args=(
                    sess,
                    coord,
                    writer,
                    env_name,
                    i,
                    self.model,
                    discount
                ))
                t.start()
                worker_threads.append(t)

                # Stagger threads to decorrelate experience
                time.sleep(i)

            t = threading.Thread(target=save_worker, args=(sess, coord, self))
            t.start()
            worker_threads.append(t)

            coord.join(worker_threads)

    def perform(self, sess, env, model, memory):
        """
        Perform action according to policy pi(a | s)
        """
        *probs, value = sess.run(
            model.model.outputs,
            memory.build_single_feed(model.model.inputs)
        )

        # Remove batch dimension
        value = value[0][0]

        # Sample an action from an action probability distribution output
        action = []

        for p in probs:
            p = p[0]
            if len(p) == 1:
                # Must be a binary probability
                action.append(round(np.random.random()))
            else:
                action.append(np.random.choice(len(p), p=p))

        flatten_action = action[0] if len(action) == 1 else action
        next_state, reward, terminal, info = env.step(flatten_action)
        next_state = self.preprocess(env, next_state)
        return value, action, next_state, reward, terminal

    def train_thread(self, sess, coord, writer, env_name, num, model, gamma):
        # Thread setup
        env = gym.make(env_name)

        episode_count = 0

        # Reset per-episode vars
        terminal = False
        total_reward = 0
        step_count = 0
        # Each memory corresponds to one input.
        memory = Memory(self.preprocess(env, env.reset()), self.time_steps)

        print("Running worker " + str(num))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # Run a training batch
                t = 0
                t_start = t

                # Batched based variables
                state_batches = [[] for _ in model.model.inputs]
                actions = []
                rewards = []
                values = []

                while not (terminal or ((t - t_start) == self.batch_size)):
                    value, action, next_state, reward, terminal = self.perform(
                        sess, env, model, memory
                    )

                    # Bookkeeping
                    for i, state in enumerate(memory.to_states()):
                        state_batches[i].append(state)

                    memory.remember(next_state)
                    actions.append(action)
                    values.append(value)
                    rewards.append(reward)

                    total_reward += reward
                    step_count += 1
                    t += 1

                if terminal:
                    reward = 0
                else:
                    # Bootstrap from last state
                    reward = sess.run(
                        model.value,
                        memory.build_single_feed(model.model.inputs)
                    )[0][0]

                # Here we take the rewards and values from the exp, and use them to
                # generate the advantage and discounted returns.
                # The advantage function uses "Generalized Advantage Estimation"
                discounted_rewards = discount(rewards, gamma, reward)
                value_plus = np.array(values + [reward])
                advantages = discount(rewards + gamma * value_plus[1:] - value_plus[:-1], gamma)

                # Train network
                v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                        model.value_loss,
                        model.policy_loss,
                        model.entropy,
                        model.grad_norms,
                        model.var_norms,
                        model.train
                    ],
                     {
                        **dict(zip(model.model.inputs, state_batches)),
                        **dict(zip(model.actions, zip(*actions))),
                        **
                        {
                            model.target_v: discounted_rewards,
                            model.advantages: advantages
                        }
                    }
                )

                if terminal:
                    # Record metrics
                    writer.add_summary(
                        make_summary({
                            'rewards': total_reward,
                            'lengths': step_count,
                            'value_loss': v_l,
                            'policy_loss': p_l,
                            'entropy_loss': e_l,
                            'grad_norm': g_n,
                            'value_norm': v_n,
                            'mean_values': np.mean(values)
                        }),
                        episode_count
                    )

                    episode_count += 1

                    # Reset per-episode counters
                    terminal = False
                    total_reward = 0
                    step_count = 0
                    # Each memory corresponds to one input.
                    memory = Memory(self.preprocess(env, env.reset()), self.time_steps)

    def run(self, sess, env):
        memory = Memory(self.preprocess(env, env.reset()), self.time_steps)
        total_reward = 0
        terminal = False

        while not terminal:
            value, action, next_state, reward, terminal = self.perform(
                sess, env, self.model, memory
            )

            total_reward += reward
            memory.remember(next_state)
