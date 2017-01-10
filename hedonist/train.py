import config
import envs
import agents
import stats


def train(config):
    train_stats = stats.Stats(config, False)
    eval_stats = stats.Stats(config, True)
    train_env = envs.Atari(config, monitor=True, monitor_name='train')
    eval_env = envs.Atari(config, monitor=True, monitor_name='eval')
    num_actions = train_env.num_actions

    train_agent = agents.DeepQLearner(
        config,
        num_actions,
        train_env,
        train_stats,
        eval_env,
        eval_stats
    )

    train_agent.train(config['training_steps'])


if __name__ == '__main__':
    config = config.get_config()
    train(config)