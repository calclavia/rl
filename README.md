# Reinforcement Learning Library
This repository aims to contain the latest reinforcement learning algorithms
implemented using Tensorflow, Keras and OpenAI Gym.

Currently, A3C has been implemented.

## Requirements
- Python 3.5

```
pip install -r requirements.txt
```

## Usage
```
agent = A3CAgent(num_actions, lambda: model)
agent.train(env_name)
```

Tensorboard Logging
```
tensorboard --logdir=out --reload_interval=2
```

Sources:
- https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
