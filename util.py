import keras
import numpy as np

def policy_loss(advantage):
    def categorical_crossentropy(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        return advantage * K.categorical_crossentropy(y_pred, y_true)
    return categorical_crossentropy

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
    return (x - np.mean(x)) / np.std(x)
