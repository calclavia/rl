from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
import numpy as np

def discount_rewards(rewards, discount):
    """ Takes an array of rewards and compute array of discounted reward """
    discounted_r = np.zeros_like(rewards)
    current = 0

    for t in reversed(range(len(rewards))):
        current = current * discount + rewards[t]
        discounted_r[t] = current

    return discounted_r

def build_rnn(input_shape, time_steps, num_h=20, num_outputs=2):
    # Build Network
    inputs = Input(shape=(time_steps,) + input_shape, name='Input')
    x = LSTM(num_h, activation='relu', name='Hidden')(inputs)
    outputs = Dense(num_outputs, activation='softmax', name='Output')(x)

    model = Model(inputs, outputs)

    # Compile for regression task
    model.compile(
        optimizer=RMSprop(lr=1e-3, clipvalue=1),
        loss='categorical_crossentropy'
    )

    return model

def z_score(x):
    # z-score the rewards to be unit normal (variance control)
    return (x - np.mean(x)) / np.std(x)
