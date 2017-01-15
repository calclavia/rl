# Reinforcement Learning Agents
This repository aims to contain the latest reinforcement learning algorithms
implemented using Tensorflow, Keras and OpenAI Gym.

Currently, A3C has been implemented.

Based on:
- https://github.com/sherjilozair/dqn
- https://github.com/tatsuyaokubo/dqn/
- http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html
- https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2

## Requirements
- Python 3.5

```
pip install -r requirements.txt
```

## Usage
```
with tf.device("/cpu:0"):
  agent = A3CAgent(num_actions, lambda: model)
  agent.train(env_name)
```
