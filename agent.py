import time

class Agent:
    def __init__(self, ob_space, action_space):
        self.ob_space = ob_space
        self.action_space = action_space

    def run(self, env, num_episodes, render=False, learn=True):
        """
        Fits this agent to the environment
        """
        mean_reward = None

        for self.num_ep in range(num_episodes):
            mean_reward = self.run_episode(env, mean_reward, render, learn)

    def run_episode(self, env, mean_reward, render=False, learn=True):
        observation = env.reset()
        done = False
        total_reward = 0
        t = time.time()

        while not done:
            if render:
                env.render()
            # Choose an action
            action = self.forward(observation)
            # Perform action
            observation, reward, done, info = env.step(action)

            if learn:
                # Observe results of chosen action
                self.backward(reward, done)
            total_reward += reward

        if mean_reward is None:
            mean_reward = total_reward

        mean_reward = mean_reward * 0.99 + total_reward * 0.01

        print('Episode {}: Reward={} ({}) Time={}'.format(
            self.num_ep,
            total_reward,
            mean_reward,
            time.time() - t
        ))

        return mean_reward

    def forward(self, observation):
        pass

    def backward(self, reward, terminal):
        pass
