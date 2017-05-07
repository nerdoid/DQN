import config
import envs
import agents
import stats


def demo(config):
    eval_env = envs.Atari(
        config,
        monitor=True,
        monitor_freq=1,
        monitor_name='demo'
    )
    num_actions = eval_env.num_actions
    eval_agent = agents.DeepQLearner(
        config,
        num_actions,
        None,
        None,
        eval_env,
        None,
        is_demo=True
    )
    eval_agent.evaluate(
        config['num_eval_episodes'],
        config['max_steps_per_eval_episode']
    )


if __name__ == '__main__':
    config = config.get_config()
    demo(config)
