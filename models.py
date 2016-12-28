from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model

def simple_rnn(input_shape, time_steps, num_h):
    # Build Network
    inputs = Input(shape=(time_steps,) + input_shape, name='input')
    x = LSTM(num_h, activation='relu', name='hidden1')(inputs)
    x = Dropout(0.25)(x)
    return Model(inputs, x)
