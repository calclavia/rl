from keras import backend as K
import numpy as np
from gym import spaces

def discount_rewards(rewards, discount):
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
