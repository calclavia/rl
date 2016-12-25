import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque

class DQNAgent:
    def __init__(self,
                 state_shape,
                 num_actions,
                 start_epsilon=1,
                 end_epsilon=0.1,
                 anneal_steps=1000000,
                 batch_size=32,
                 discount=0.99,
                 memory=400000,
                 target_update_interval=10000,
                 initial_replay_size=10000,
                 train_interval=4):

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount = discount
        self.memory = memory
        self.target_update_interval = target_update_interval
        self.initial_replay_size = initial_replay_size
        self.train_interval = train_interval

        # Epsilon
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.annealing = (start_epsilon - end_epsilon) / anneal_steps

        self.i = 0

        # Experience replay memory
        self.replay_memory = deque()

        # Create q network model
        self.model = self.build_network()
        model_weights = self.model.trainable_weights

        # Create target network model
        self.target_model = self.build_network()
        target_weights = self.target_model.trainable_weights

        # Define target network update operation
        self.update_target_model = [target_weights[i].assign(model_weights[i]) for i in range(len(target_weights))]

        # Update target weights
        sess = K.get_session()
        sess.run(self.update_target_model)

    def build_network(self):
        """
        Build the DQN model
        """
        model = Sequential()
        model.add(Dense(20, input_shape=self.state_shape, activation='tanh'))
        model.add(Dense(20, activation='tanh'))
        model.add(Dense(self.num_actions))

        # Compile for regression task
        model.compile(
            optimizer=RMSprop(lr=1e-4, clipvalue=1),
            loss='mean_squared_error'
        )

        return model

    def choose(self, state):
        """
        The agent observes a state and chooses an action.
        """
        predictions = self.model.predict(np.array([state]))[0]

        # epsilon greedy exploration-exploitation
        if np.random.random() < self.epsilon:
            # Take a random action
            action = np.random.randint(self.num_actions)
        else:
            # Get q values for all actions in current state
            # Take the greedy policy (choose action with largest q value)
            action = np.argmax(predictions)

        # Epsilon annealing
        if self.epsilon > self.end_epsilon:
            self.epsilon -= self.annealing

        return action, state, predictions

    def observe(self, state, action, reward, next_state, terminal):
        """
        Observe reward and the new state from performing an action
        """
        # The target vector should equal the output for all elements
        # except the element whose action we chose.
        a_hot = np.array([int(i == action) for i in range(self.num_actions)])

        # Store experience in memory
        self.replay_memory.append((state, a_hot, reward, next_state, 0 if terminal else 1))

        # Eject old memory
        if len(self.replay_memory) > self.memory:
            self.replay_memory.popleft()

    def learn(self):
        """
        Learns from replay memory
        """
        loss = 0

        if self.i > self.initial_replay_size and self.i % self.train_interval:
            # Sample minibatch data
            sample_size = min(self.batch_size, len(self.replay_memory))
            minibatch = random.sample(self.replay_memory, sample_size)
            inputs = []
            targets = []

            for state, a_hot, reward, next_state, terminal in minibatch:
                # Build inputs and targets to fit the model to
                target_val = reward + terminal * self.discount * np.max(self.target_model.predict(np.array([next_state])))
                target = self.model.predict(np.array([next_state]))[0] * (1 - a_hot) + target_val * a_hot
                inputs.append(state)
                targets.append(target)

            inputs = np.array(inputs)
            targets = np.array(targets)

            # TODO: Inefficient recomputation of outputs?
            self.model.fit(inputs, targets, verbose=0)
            loss = self.model.evaluate(inputs, targets, verbose=0)

        if self.i % self.target_update_interval:
            sess = K.get_session()
            sess.run(self.update_target_model)

        self.i += 1
        return loss

    def load(self):
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print("Loading weights from {}.h5".format(self.save_name))
        except:
            print("Training a new model")

    def save(self):
        # TODO: Decouple save logic?
        # Save model after several intervals
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)
