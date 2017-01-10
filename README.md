# Hedonist

This project is a collection of exploratory deep reinforcement learning algorithms implemented in TensorFlow and the OpenAI Gym. It currently includes just the [DQN](https://deepmind.com/research/dqn/) and [Double DQN](https://arxiv.org/abs/1509.06461) algorithms. As I learn other RL algorithms, I will add them.

I'm testing the limits of how much functionality can be implemented in TensorFlow exclusively rather than the usual libraries like NumPy. The motivation is:

1. Easy device placement - TensorFlow makes this super simple.
2. Future proofing - As GPU RAM goes up, we can keep more GPU-side and reduce fetching delays.
3. Master the computational graph - Immerse myself in the computational graph paradigm to better understand it.

As such, a novel aspect of Hedonist's code is that the DQN's replay memory is implemented using TensorFlow ops. This may very well turn out to be a terrible idea. But I won't know until I run some benchmarks.

# Requirements

Python 3

TensorFlow 0.12

# Getting Started

## Training

To train an agent (on Breakout by default):
```bash
> python hedonist/train.py
```

All summaries, videos, and checkpoints will go to the `results` directory.

## Demos

You can record vidoes using a trained model by running:
```bash
> python hedonist/demo.py
```

# Configuration

To customize a training or demo run (for example to use a different game), change the available settings in `hedonist/config.py`.
