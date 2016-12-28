from copy import deepcopy
from agent import Agent
from a2c import A2CAgent
from collections import deque
import threading
import numpy as np

class A3CAgent(Agent):
    """
    Adapts the agent to run in an async environment.
    """

    def __init__(self, ob_space, action_space, exp, num_agents=4):
        super().__init__(ob_space, action_space, exp)

        self.agents = [A2CAgent(ob_space, action_space, deepcopy(exp)) for _ in range(num_agents)]

    def compile(self, model):
        self.agents[0].compile(model)

        for agent in self.agents:
            agent.critic.model = self.agents[0].critic.model
            agent.actor.model = self.agents[0].actor.model
            agent.actor.trainer = self.agents[0].actor.trainer


    def run(self, env, num_episodes, render=False, learn=True):
        # Simulate parallelism
        for agent in self.agents:
            agent.total_rewards = deque(maxlen=100)
            agent.num_ep = 0

        super().run(env, num_episodes, render, learn)

        """
        processes = []
        for agent in self.agents:
            # Copy the environment
            run = lambda: agent.run(deepcopy(env), num_episodes, render, learn)
            p = threading.Thread(target=run)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        """
    def run_episode(self, env, render=False, learn=True):
        total_reward = 0
        mean_reward = []

        # Simulate parallelism
        for agent in self.agents:
            agent.num_ep += 1
            agent.run_episode(env, render, learn)
            total_reward += agent.total_rewards[-1]
            mean_reward += agent.total_rewards

        print('== Episode {}: Reward={} ({}) =='.format(
            self.num_ep,
            total_reward,
            np.mean(mean_reward)
        ))
