import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque

class Agent:
    def __init__(self,
                 state_shape=None,
                 num_actions=1,
                 start_epsilon=1,
                 end_epsilon=0.1,
                 anneal_steps=1000000,
                 mbsz=32,
                 discount=0.99,
                 memory=400000,
                 target_update_interval=10000):

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.mbsz = mbsz
        self.discount = discount
        self.memory = memory
        self.target_update_interval = target_update_interval

        # Epsilon
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.annealing = (start_epsilon - end_epsilon) / anneal_steps

        self.i = 0

        # Experience replay (s, a, r, s') tuple
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
        model.add(Dense(10, input_shape=self.state_shape, activation='relu'))
        model.add(Dense(self.num_actions))

        # Compile for regression task
        model.compile(
            optimizer=RMSprop(lr=1e-4),
            loss='mean_squared_error'
        )

        return model

    """
    def build_loss(self):
        ""
        Build the loss function.

        Loss function takes in an instance of replay memory for one training
        data in the form of (s, a, r, s')
        ""
        # Current state vector
        S = K.placeholder(shape=self.state_shape)
        # The action taken (int scalar)
        a = K.placeholder(shape=(), dtype='int32')
        # The reward received (float scalar)
        r = K.placeholder(shape=(), dtype='float32')
        # Next state vector
        NS = K.placeholder(shape=self.state_shape)

        def dqn_loss(y_true, y_pred):



        # Output tensor of the model with zero gradient for next state
        next_q_value = K.stop_gradient(self.model(NS))

        # Compute the target for the action taken.
        # All other targets = y (no change)
        action_hot = K.one_hot(a[0], self.num_actions)
        print(action_hot, a, self.num_actions)
        target_q = r + self.discount * action_hot * K.max(next_q_value)
        print(target_q)
        loss = mean_squared_error(target_q, q_value)
        print(loss)
        opt = RMSprop(0.0001)
        updates = opt.get_updates(self.model.trainable_weights, [], loss)
        self.train_fn = K.function([S, a, r, NS], [loss], updates)
    """
    def choose(self, state):
        """
        The agent observes a state and chooses an action.
        """
        # epsilon greedy exploration-exploitation
        if np.random.random() < self.epsilon:
            # Take a random action
            action = np.random.randint(self.num_actions)
        else:
            # Get q values for all actions in current state
            # Take the greedy policy (choose action with largest q value)
            action = np.argmax(self.model.predict(np.array([state])))

        # Epsilon annealing
        if self.epsilon > self.end_epsilon:
            self.epsilon -= self.annealing

        return action, state

    def observe(self, state, action, reward, next_state):
        """
        Observe reward and the new state from performing an action
        """
        # Store experience in memory
        self.replay_memory.append((state, action, reward, next_state))

        # Eject old memory
        if len(self.replay_memory) > self.memory:
            self.replay_memory.popleft()

    def learn(self, terminal):
        """
        Learns from replay memory
        """
        # TODO: Not an efficient way. Many recomputations performed

        # Sample minibatch data
        sample_size = min(self.mbsz, len(self.replay_memory))
        minibatch = random.sample(self.replay_memory, sample_size)

        # Build inputs and targets to fit the model to
        inputs = []
        targets = []

        for state, action, reward, next_state in minibatch:
            # Create data and labels by extracting from replay memory
            if terminal:
                target_val = reward
            else:
                target_val = reward + self.discount * np.max(self.target_model.predict(np.array([next_state])))

            # The target vector should equal the output for all elements
            # except the element whose action we chose.
            output = self.model.predict(np.array([state]))[0]
            a_hot = np.array([int(i == action) for i in range(self.num_actions)])
            target = output * (1 - a_hot) + target_val * a_hot
            inputs.append(state)
            targets.append(target)

        # TODO: Inefficient recomputation of outputs
        self.model.fit(np.array(inputs), np.array(targets), verbose=0)

        if self.i % self.target_update_interval:
            sess = K.get_session()
            sess.run(self.update_target_model)

        self.i += 1

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
