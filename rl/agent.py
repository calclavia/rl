"""
General interface for an agent
"""
class Agent:
    """
    Represents an agent
    """
    def compile(self, sess):
        """
        Compiles the agent, setting up all the models and ops.
        """
        pass

    def train(self, env_builder):
        """
        Trains the agent on an environment
        """
        pass

    def run(self, env_builder):
        """
        Runs the agent in an environment
        """
        pass
