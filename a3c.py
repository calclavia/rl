import numpy as np
import tensorflow as tf
import os
import gym
import threading
import multiprocessing
import time
from collections import deque
from gym import spaces

from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from .util import *

class ACModel:
    def __init__(self, model_builder, num_actions, scope, beta):
        self.scope = scope
        self.num_actions = num_actions
        # Entropy weight
        self.beta = beta

        with tf.variable_scope(self.scope):
            self.inputs, x = model_builder()
            # Output layers for policy and value estimations
            self.policy = Dense(num_actions, activation='softmax', name='policy_output')(x)
            self.value = Dense(1, activation='linear', name='value_output')(x)
            self.model = Model(self.inputs, [self.policy, self.value])

    def compile(self, optimizer, grad_clip):
        # Only the worker network need ops for loss functions and gradient updating.
        with tf.variable_scope(self.scope):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            actions_hot = tf.one_hot(self.actions, self.num_actions)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            responsible_outputs = tf.reduce_sum(self.policy * actions_hot, [1])

             # Value loss (Mean squared error)
            self.value_loss = tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            # Entropy regularization
            self.entropy = -tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0)))
            # Policy loss
            self.policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * self.advantages)
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
class Memory:
    """
    Represents the memory of the agent.
    The agent by default stores only the current time step, but is capable
    of holding memory of previos time steps for training RNNs.
    """

    def __init__(self, init_state, time_steps):
        self._memory = []
        self.time_steps = time_steps

        # Handle non-tuple states
        if not isinstance(init_state, tuple):
            self.is_tuple = False
            init_state = (init_state,)
        else:
            self.is_tuple = True

        for input_state in init_state:
            # lookback buffer
            temporal_memory = deque(maxlen=max(time_steps, 1))
            # Fill temporal memory with zeros
            while len(temporal_memory) < time_steps - 1:
                temporal_memory.appendleft(np.zeros_like(input_state))

            temporal_memory.append(input_state)
            self._memory.append(temporal_memory)

    def remember(self, state):
        if not self.is_tuple:
            state = (state,)

        for i, input_state in enumerate(state):
            self._memory[i].append(input_state)

    def to_states(self):
        """ Returns a state per input """
        if self.time_steps == 0:
            # No time_steps = not recurrent
            return [m[0] for m in self._memory]
        else:
            return [list(m) for m in self._memory]

    def build_single_feed(self, inputs):
        if self.time_steps == 0:
            # No time_steps = not recurrent
            return {i: list(m) for i, m in zip(inputs, self._memory)}
        else:
            return {i: [list(m)] for i, m in zip(inputs, self._memory)}

class A3CAgent:
    def __init__(self,
                 num_actions,
                 model_builder,
                 time_steps=0,
                 preprocess=lambda e, x: x,
                 model_path='out/model',
                 entropy_factor=1e-2,
                 batch_size=32):
        self.num_actions = num_actions
        self.model_builder = model_builder
        self.time_steps = time_steps
        self.model_path = model_path
        self.entropy_factor = entropy_factor
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.save_count = 0
        # Generate global network
        self.model = ACModel(model_builder, num_actions, 'global', entropy_factor)
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
        print(self.model.model.summary())

        with tf.Session() as sess:
            workers = []

            # Create worker classes
            for i in range(num_workers):
                name = 'worker_' + str(i)
                model = ACModel(
                    self.model_builder,
                    self.num_actions,
                    name,
                    self.entropy_factor
                )
                model.compile(optimizer, grad_clip)
                sync = update_target_graph('global', name)
                workers.append((name, model, sync))

            # Initialize variables
            sess.run(tf.global_variables_initializer())
            try:
                self.load(sess)
                print('Loading last saved session')
            except:
                print('Starting new session')

            coord = tf.train.Coordinator()
            worker_threads = []

            for i, (name, model, sync) in enumerate(workers):
                writer = tf.summary.FileWriter(summary_path + name, sess.graph, flush_secs=2)
                t = threading.Thread(target=self.train_thread, args=(
                    sess,
                    coord,
                    writer,
                    env_name,
                    i,
                    model,
                    sync,
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

    def train_thread(self, sess, coord, writer, env_name, num, model, sync, gamma):
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
                # Sync local network with global network
                sess.run(sync)

                # Run a training batch
                t = 0
                t_start = t

                # Batched based variables
                state_batches = [[] for _ in model.inputs]
                actions = []
                rewards = []
                values = []

                while not (terminal or ((t - t_start) == self.batch_size)):
                    # Perform action according to policy pi(a_t | s_t)
                    probs, value = sess.run(
                        [model.policy, model.value],
                         memory.build_single_feed(model.inputs)
                    )

                    # Remove batch dimension
                    probs = probs[0]
                    value = value[0]
                    # Sample an action from an action probability distribution output
                    action = np.random.choice(len(probs), p=probs)

                    next_state, reward, terminal, info = env.step(action)
                    next_state = self.preprocess(env, next_state)

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
                        memory.build_single_feed(model.inputs)
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
                        **dict(zip(model.inputs, state_batches)),
                        **
                        {
                            model.target_v: discounted_rewards,
                            model.actions: actions,
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
            # Perform action according to policy pi(a_t | s_t)
            probs, value = sess.run(
                [self.model.policy, self.model.value],
                memory.build_single_feed(self.model.inputs)
            )

            # Remove batch dimension
            probs = probs[0]
            value = value[0]
            # Sample an action from an action probability distribution output
            action = np.random.choice(len(probs), p=probs)
            state, reward, terminal, info = env.step(action)
            total_reward += reward
            memory.remember(self.preprocess(env, state))
