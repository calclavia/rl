import numpy as np
import tensorflow as tf
import os
import threading
import multiprocessing
import time

from .a3c_model import ACModel
from .util import *
from .memory import Memory
from .agent import Agent

class ACAgentRunner(Agent):

    def __init__(self, model, memory, preprocess, batch_size):
        self.model = model
        self.memory = memory
        self.preprocess = preprocess
        self.batch_size = batch_size

    def perform(self, sess, env):
        """
        Perform action according to policy pi(a | s)
        """
        *probs, value = sess.run(
            self.model.model.outputs,
            self.memory.build_single_feed(self.model.model.inputs)
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

    def train(self, sess, coord, env_builder, writer, gamma):
        try:
            # Thread setup
            env = env_builder()

            episode_count = 0

            # Reset per-episode vars
            terminal = False
            total_reward = 0
            step_count = 0

            # Each memory corresponds to one input.
            self.memory.reset(self.preprocess(env, env.reset()))

            print("Training ACAgentRunner...")

            while not coord.should_stop():
                # Run a training batch
                t = 0
                t_start = t

                # Batched based variables
                state_batches = [[] for _ in self.model.model.inputs]
                actions = []
                rewards = []
                values = []

                while not (terminal or ((t - t_start) == self.batch_size)):
                    value, action, next_state, reward, terminal = self.perform(
                        sess, env
                    )

                    # Bookkeeping
                    for i, state in enumerate(self.memory.to_states()):
                        state_batches[i].append(state)

                    self.memory.remember(next_state)
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
                        self.model.value,
                        self.memory.build_single_feed(
                            self.model.model.inputs)
                    )[0][0]

                # Here we take the rewards and values from the exp, and use them to
                # generate the advantage and discounted returns.
                # The advantage function uses "Generalized Advantage
                # Estimation"
                discounted_rewards = discount(rewards, gamma, reward)
                value_plus = np.array(values + [reward])
                advantages = discount(
                    rewards + gamma * value_plus[1:] - value_plus[:-1], gamma)

                # Train network
                v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                    self.model.value_loss,
                    self.model.policy_loss,
                    self.model.entropy,
                    self.model.grad_norms,
                    self.model.var_norms,
                    self.model.train
                ],
                    {
                    **dict(zip(self.model.model.inputs, state_batches)),
                        **dict(zip(self.model.actions, zip(*actions))),
                        **
                    {
                            self.model.target_v: discounted_rewards,
                            self.model.advantages: advantages
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
                    self.memory.reset(self.preprocess(env, env.reset()))
        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)

    def run(self, sess, env):
        self.memory.reset(self.preprocess(env, env.reset()))
        total_reward = 0
        terminal = False

        while not terminal:
            value,\
            action,\
            next_state,\
            reward,\
            terminal = self.perform(sess, env)

            total_reward += reward
            self.memory.remember(next_state)

# TODO: Refactor to async coordinator?
class A3CAgent(Agent):
    # TODO: Refactor these hyperparameters to one object

    def __init__(self,
                 model_builder,
                 time_steps=0,
                 preprocess=lambda e, x: x,
                 model_path='out/model',
                 num_workers=multiprocessing.cpu_count(),
                 entropy_factor=1e-2,
                 batch_size=32):
        self.model_builder = model_builder
        self.time_steps = time_steps
        self.model_path = model_path
        self.entropy_factor = entropy_factor
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.save_count = 0
        self.model = ACModel(model_builder, entropy_factor)
        self.saver = tf.train.Saver(max_to_keep=5)

        # Create agents
        self.agents = []

        for i in range(num_workers):
            self.add_agent()

    def add_agent(self, Agent=ACAgentRunner):
        self.agents.append(Agent(
            self.model,
            Memory(self.time_steps),
            self.preprocess,
            self.batch_size
        ))

    # TODO: Not SRP. Agent shouldn't handle model saving.
    def load(self, sess):
        self.model.model.load_weights(self.model_path + '/model.h5')

    def save(self, sess):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.model.save_weights(self.model_path + '/model_' + str(self.save_count) + '.h5')
        self.save_count += 1

    def compile(self,
                sess,
                grad_clip=50.,
                optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):
        self.model.compile(optimizer, grad_clip)
        print(self.model.model.summary())

        # Initialize variables
        sess.run(tf.global_variables_initializer())

    def train(self,
              sess,
              env_builder,
              summary_path='out/summary/',
              discount=.99):
        """
        Starts training.
        Return: The coordinator for all the threads
        """
        print('Training model')

        coord = tf.train.Coordinator()

        for i, agent in enumerate(self.agents):
            name = 'worker_' + str(i)
            writer = tf.summary.FileWriter(
                summary_path + name, sess.graph, flush_secs=2)

            t = threading.Thread(
                target=agent.train,
                args=(
                    sess,
                    coord,
                    env_builder,
                    writer,
                    discount
                )
            )

            t.start()
            coord.register_thread(t)

            # Stagger threads to decorrelate experience
            time.sleep(1)

        # Create thread that auto-saves
        t = threading.Thread(target=save_worker, args=(sess, coord, self))
        t.start()
        coord.register_thread(t)

        return coord

    def run(self, sess, env):
        # Pick the first agent to run the environment
        self.agents[0].run(sess, env)
