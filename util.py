from keras import backend as K
import numpy as np
import tensorflow as tf
from gym import spaces

def discount(rewards, discount):
    """ Takes an array of rewards and compute array of discounted reward """
    discounted_r = np.zeros_like(rewards)
    current = 0

    for t in reversed(range(len(rewards))):
        current = current * discount + rewards[t]
        discounted_r[t] = current

    return discounted_r

def z_score(x):
    # z-score the rewards to be unit normal (variance control)
    std = np.std(x)

    if std != 0:
        return (x - np.mean(x)) / std
    return x - np.mean(x)

def space_to_shape(space):
    if isinstance(space, spaces.Discrete):
        # One hot vectors of states
        return (space.n,)

    if isinstance(space, spaces.Tuple):
        return (len(space.spaces),)

    return space.shape

def action_to_shape(space):
    return space.n if isinstance(space, spaces.Discrete) else space.shape

def one_hot(index, size):
    return [1 if index == i else 0 for i in range(size)]

def summary_writer(prefix, writer):
    def write(agent):
        summary = tf.Summary()

        for name, values in agent.metrics.items():
            summary.value.add(tag=prefix + '/' + name, simple_value=float(values[-1]))

        writer.add_summary(summary, agent.episode_count)
        writer.flush()

    return write

def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
