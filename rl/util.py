from keras import backend as K
import time
import numpy as np
import tensorflow as tf
from gym import spaces

def discount(rewards, discount, current=0):
    """ Takes an array of rewards and compute array of discounted reward """
    discounted_r = np.zeros_like(rewards)

    for t in reversed(range(len(rewards))):
        current = current * discount + rewards[t]
        discounted_r[t] = current

    return discounted_r

def make_summary(data, prefix=''):
    if prefix != '':
        prefix += '/'

    summary = tf.Summary()
    for name, value in data.items():
        summary.value.add(tag=prefix + name, simple_value=float(value))

    return summary

def save_worker(sess, coord, agent):
    while not coord.should_stop():
        time.sleep(60)
        agent.save(sess)

def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def track(env):
    """
    Wraps a Gym environment to keep track of the results of step calls visited.
    """
    step = env.step
    def step_override(*args, **kwargs):
        result = step(*args, **kwargs)
        env.step_cache.append(result)
        env.total_reward += result[1]
        return result
    env.step = step_override

    reset = env.reset
    def reset_override(*args, **kwargs):
        env.total_reward = 0
        env.step_cache = []
        return reset(*args, **kwargs)
    env.reset = reset_override

    return env
