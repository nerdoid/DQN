import argparse
import config
import envs
import agents
import stats


def train(config, run_name):
    train_stats = stats.Stats(config, run_name, False)
    eval_stats = stats.Stats(config, run_name, True)
    train_env = envs.Atari(
        config, monitor=True, monitor_name='train', run_name=run_name
    )
    eval_env = envs.Atari(
        config,
        monitor=True,
        monitor_name='eval',
        monitor_freq=10,
        run_name=run_name
    )
    num_actions = train_env.num_actions

    train_agent = config['agent_type'](
        config,
        num_actions,
        train_env,
        train_stats,
        eval_env,
        eval_stats,
        run_name=run_name
    )

    train_agent.train(config['training_steps'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        help='Name this run for display in TensorBoard.',
        required=True
    )
    args = parser.parse_args()

    config = config.get_config()

    train(config, args.name)
