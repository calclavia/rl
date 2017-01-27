"""
A simple example to run the DQN algorithm on a toy example.
"""
import gym
import tensorflow as tf
from rl import DQNAgent
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.models import Model

env_name = 'CartPole-v0'
num_actions = 2

def make_model():
    i = Input((4,))
    x = i
    x = Dense(128, activation='relu')(x)
    policy = Dense(num_actions, activation='softmax')(x)
    value = Dense(1, activation='linear')(x)
    return Model([i], [policy, value])

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = DQNAgent(make_model)
    agent.compile(sess)
    agent.train(sess, lambda: gym.make('CartPole-v0')).join()
