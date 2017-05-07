# DQN

This project is an implementation of DeepMind's DQN algorithm and its associated bag of tricks. It relies on TensorFlow and the OpenAI Gym. It currently includes the [DQN](https://deepmind.com/research/dqn/) and [Double DQN](https://arxiv.org/abs/1509.06461) algorithms. Most notably, prioritized experience replay is not yet implemented.

# Requirements

Python 3 or greater

TensorFlow 1.0 or greater

# Getting Started

## Training

To train an agent (on Breakout by default):
```bash
> python dqn/train.py --name [name_for_this_run]
```

All summaries, videos, and checkpoints will go to the `results` directory.

## Demos

You can record vidoes using a trained model by running:
```bash
> python dqn/demo.py
```

# Configuration

To customize a training or demo run (for example to use a different game), change the available settings in `dqn/config.py`.


# Analysis

Since every run has a name, TensorBoard summaries are automatically written to a corresponding subdirectory under `results/stats`. Algorithmic variations can then be compared with graphical overlays in TensorBoard:
```bash
> tensorboard --logdir=results/stats
```


## Sample Stats

Running vanilla DQN on OpenAI Gym environment BreakoutDeterministic-v3

<img src="https://raw.githubusercontent.com/nerdoid/DQN/master/assets/average_score_per_episode.png?raw=true" alt="Average Score" title="Average Score" width="307" height="207">
<img src="https://raw.githubusercontent.com/nerdoid/DQN/master/assets/max_score_per_episode.png?raw=true" alt="Average Score" title="Average Score" width="307" height="207">
