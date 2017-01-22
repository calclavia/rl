import numpy as np
from collections import deque

class Memory:
    """
    Represents the memory of the agent.
    The agent by default stores only the current time step, but is capable
    of holding memory of previos time steps for training RNNs.
    """

    def __init__(self, time_steps):
        self.time_steps = time_steps

    def reset(self, init_state):
        self._memory = []

        # Handle non-tuple states
        if not isinstance(init_state, tuple):
            self.is_tuple = False
            init_state = (init_state,)
        else:
            self.is_tuple = True

        for input_state in init_state:
            # lookback buffer
            temporal_memory = deque(maxlen=max(self.time_steps, 1))
            # Fill temporal memory with zeros
            while len(temporal_memory) < self.time_steps - 1:
                temporal_memory.appendleft(np.zeros_like(input_state))

            temporal_memory.append(input_state)
            self._memory.append(temporal_memory)

    def remember(self, state):
        if not self.is_tuple:
            state = (state,)

        for i, input_state in enumerate(state):
            self._memory[i].append(input_state)

    def to_states(self):
        """ Returns a state per input """
        if self.time_steps == 0:
            # No time_steps = not recurrent
            return [m[0] for m in self._memory]
        else:
            return [list(m) for m in self._memory]

    def build_single_feed(self, inputs):
        if self.time_steps == 0:
            # No time_steps = not recurrent
            return {i: list(m) for i, m in zip(inputs, self._memory)}
        else:
            return {i: [list(m)] for i, m in zip(inputs, self._memory)}
