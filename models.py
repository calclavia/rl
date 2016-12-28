from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model

def simple_rnn(input_shape, time_steps, num_h):
    # Build Network
    inputs = x = Input(shape=(time_steps,) + input_shape, name='input')
    x = LSTM(num_h, activation='relu', name='hidden1')(x)
    x = Dropout(0.25)(x)
    x = Dense(num_h, activation='relu', name='hidden2')(x)
    x = Dropout(0.25)(x)
    x = Dense(num_h, activation='relu', name='hidden3')(x)
    x = Dropout(0.25)(x)
    return Model(inputs, x)
