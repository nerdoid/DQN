"""Configuration data for training or evaluating a reinforcement learning
agent.
"""
import agents


def get_config():
    config = {
        'game': 'BreakoutDeterministic-v3',
        'agent_type': agents.DeepQLearner,
        'history_length': 4,
        'training_steps': 50000000,
        'training_freq': 4,
        'num_eval_episodes': 30,
        'max_steps_per_eval_episode': 135000,
        'eval_freq': 150000,
        'initial_replay_size': 50000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay_steps': 1000000,
        'eval_epsilon': 0.05,
        'screen_dims': (84, 84),
        'reward_processing': 'clip',
        'discount_factor': 0.99,
        'learning_rate': 0.00025,
        'rms_scale': 0.95,
        'rms_constant': 0.01,
        'error_clipping': 1.0,
        'target_update_freq': 10000,
        'memory_capacity': 1000000,
        'batch_size': 32,
        'summary_freq': 50000,
    }

    return config
