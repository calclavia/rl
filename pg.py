""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, merge
from keras.optimizers import RMSprop
import gym

inputs = Input(shape=(4,))
x = Dense(20, activation='relu')(inputs)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile for regression task
model.compile(
    optimizer=RMSprop(lr=1e-4, clipvalue=1),
    loss='categorical_crossentropy'
)

# hyperparameters
discount = 1  # discount factor for reward
batch_size = 32
render = False


def discount_rewards(rewards):
    """ Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    current = 0

    for t in reversed(range(len(rewards))):
        current = current * discount + rewards[t]
        discounted_r[t] = current

    return discounted_r

env = gym.make("CartPole-v0")

running_reward = None
num_ep = 0

input_buffer = []
target_buffer = []

while running_reward == None or running_reward < 1000:
    done = False
    reward_sum = 0
    observation = env.reset()  # reset env

    # reset array memory
    states, targets, rewards = [], [], []

    while not done:
        if render:
            env.render()

        # forward the policy network and sample an action from the returned
        # probability
        prob_dist = model.predict(np.array([observation]))[0]
        action = np.random.choice(prob_dist.size, p=prob_dist)

        # record various intermediates (needed later for backprop)
        states.append(observation)  # observation

        # grad that encourages the action that was taken to be taken (see
        # http://cs231n.github.io/neural-networks-2/#losses if confused)
        # a "fake label"
        target = np.array(
            [1. if action == i else 0. for i in range(len(prob_dist))])

        targets.append(target)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # record reward
        rewards.append(reward)

    num_ep += 1

    # all inputs, action gradients, and rewards
    targets = np.vstack(targets)
    rewards = np.vstack(rewards)

    # compute the discounted reward backwards through time
    discounted_rewards = discount_rewards(rewards)

    # z-score the rewards to be unit normal (variance control)
    # TODO: Seems like disabling this still allows it to work!
    #discounted_rewards -= np.mean(discounted_rewards)
    #discounted_rewards /= np.std(discounted_rewards)

    # modulate the gradient with advantage (PG magic happens right here.)
    # TODO: Is modulating the targe equiv? Maybe need to adjust loss func
    targets *= discounted_rewards

    # Buffer
    input_buffer += states
    target_buffer += targets.tolist()

    if num_ep % batch_size == 0:
        input_buffer = np.array(input_buffer)
        target_buffer = np.array(target_buffer)

        # TODO: Seems like epochs actually improve training speed?
        model.fit(input_buffer, target_buffer, verbose=0, nb_epoch=1)

        input_buffer = []
        target_buffer = []

    # book-keeping
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
