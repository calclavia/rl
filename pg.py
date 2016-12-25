""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym

# hyperparameters
H = 20  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 1  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
render = False

# model initialization
D = 4
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)

# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
# rmsprop memory
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

def sigmoid(x):
    # sigmoid "squashing" function to interval [0,1]
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    return I


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

env = gym.make("CartPole-v0")

running_reward = None
num_ep = 0

while running_reward == None or running_reward < 300:
    done = False
    reward_sum = 0
    observation = env.reset()  # reset env

    xs, hs, dlogps, drs = [], [], [], []  # reset array memory

    while not done:
        if render:
            env.render()

        # preprocess the observation
        x = prepro(observation)

        # forward the policy network and sample an action from the returned
        # probability
        aprob, h = policy_forward(x)
        action = 1 if np.random.uniform() > aprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state

        # grad that encourages the action that was taken to be taken (see
        # http://cs231n.github.io/neural-networks-2/#losses if confused)
        target = action  # a "fake label"

        dlogps.append(target - aprob)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # record reward (has to be done after we call step() to get reward for
        # previous action)
        drs.append(reward)
    num_ep += 1
    # stack together all inputs, hidden states, action gradients, and
    # rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient
    # estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    # modulate the gradient with advantage (PG magic happens right here.)
    epdlogp *= discounted_epr
    grad = policy_backward(eph, epdlogp)

    for k in model:
        grad_buffer[k] += grad[k]  # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if num_ep % batch_size == 0:
        for k, v in model.items():
            g = grad_buffer[k]  # gradient
            rmsprop_cache[k] = decay_rate * \
                rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / \
                (np.sqrt(rmsprop_cache[k]) + 1e-5)
            # reset batch gradient buffer
            grad_buffer[k] = np.zeros_like(v)

    # boring book-keeping
    if running_reward is None:
        running_reward = reward_sum

    running_reward = running_reward * 0.99 + reward_sum * 0.01

    print('Episode {}: Reward: {} Mean: {}'.format(
        num_ep, reward_sum, running_reward))


def test():
    done = False
    reward_sum = 0
    observation = env.reset()  # reset env
    while not done:
        env.render()

        # preprocess the observation
        x = prepro(observation)

        # forward the policy network and sample an action from the returned
        # probability
        aprob, h = policy_forward(x)
        action = 1 if np.random.uniform() > aprob else 0
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
    print('Reward: {}'.format(reward_sum))

while True:
    test()
