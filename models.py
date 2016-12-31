from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model

def rnn(input_shape, time_steps, num_h, layers):
    # Build Network
    # Dropout hyper-parameters based on Hinton's paper
    inputs = x = Input(shape=(time_steps,) + input_shape, name='input')
    x = Dropout(0.2)(x)
    x = LSTM(num_h, activation='relu', name='hidden')(x)
    x = Dropout(0.5)(x)

    for i in range(layers - 1):
        x = Dense(num_h, activation='relu', name='hidden' + str(i))(x)
        x = Dropout(0.5)(x)

    return Model(inputs, x)

def dense(input_shape, num_h, layers, dropout=0):
    # Build Network
    # Dropout hyper-parameters based on Hinton's paper
    inputs = x = Input(shape=input_shape, name='input')
    if dropout != 0:
        x = Dropout(dropout)(x)

    for i in range(layers):
        x = Dense(num_h, activation='relu', name='hidden' + str(i))(x)
        if dropout != 0:
            x = Dropout(dropout)(x)

    return inputs, x
