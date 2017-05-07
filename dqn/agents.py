import random
import numpy as np
import gym
import replay
import networks
import tensorflow as tf


class DeepQLearner():
    """Agent that executes the deep Q learning algorithm."""
    def __init__(self, config, num_actions, train_env, train_stats,
                 eval_env, eval_stats, is_demo=False, run_name=None):
        sess = tf.Session()

        self.network = self.create_network(
            config, num_actions, is_demo, run_name
        )
        self.train_env = train_env
        self.train_stats = train_stats
        self.eval_env = eval_env
        self.eval_stats = eval_stats

        self.replay = replay.Replay(
            sess,
            config['memory_capacity'],
            config['screen_dims'],
            config['history_length'],
            config['batch_size'],
            num_actions
        )

        sess.run(tf.global_variables_initializer())

        self.num_actions = num_actions
        self.history_length = config['history_length']
        self.initial_replay_size = config['initial_replay_size']
        self.training_freq = config['training_freq']
        self.eval_freq = config['eval_freq']
        self.num_eval_episodes = config['num_eval_episodes']
        self.max_steps_per_eval_episode = config['max_steps_per_eval_episode']
        self.eval_epsilon = config['eval_epsilon']
        self.epsilons = np.linspace(
            config['epsilon_start'],
            config['epsilon_end'],
            config['epsilon_decay_steps']
        )
        self.epsilon_decay_steps = config['epsilon_decay_steps']
        self.summary_freq = config['summary_freq']
        self.reward_processing = config['reward_processing']

    def create_network(self, config, num_actions, is_demo, run_name):
        return networks.QNetwork(config, num_actions, is_demo, run_name)

    def choose_action(self, epsilon):
        if random.random() >= epsilon:
            state = [self.replay.get_current_state()]
            q_values = self.network.predict(state)
            return np.argmax(q_values)
        else:
            return random.randrange(self.num_actions)

    def choose_eval_action(self, state, epsilon):
        q_values = None
        action = None

        if random.random() >= epsilon:
            prediction_state = np.expand_dims(
                np.transpose(state, [1, 2, 0]),
                axis=0
            )
            q_values = self.network.predict(prediction_state)
            action = np.argmax(q_values)
        else:
            action = random.randrange(self.num_actions)

        return action


    def start_new_episode(self, env):
        try:
            initial_state = env.reset()
            for experience in initial_state:
                self.replay.insert(
                    experience[0],
                    experience[1],
                    experience[2],
                    experience[3]
                )
            self.train_stats.increment_episode_counter()
        except gym.error.Error:
            # Just lost a life. Keep going.
            pass

    def process_reward(self, reward):
        """A chance to modify the reward before it is saved for training.
        NOTE: Should this be done elsewhere? Maybe the memory system?
        """
        if self.reward_processing == 'clip':
            return np.clip(reward, -1, 1)
        else:
            return reward

    def populate_replay(self, initial_replay_size):
        for step in range(initial_replay_size):
            action = random.randrange(self.num_actions)
            step_result = self.train_env.step(action)
            frame, raw_reward, terminal, _ = step_result
            reward = self.process_reward(raw_reward)

            self.train_stats.add_reward(reward)
            self.replay.insert(frame, action, reward, terminal)

            if terminal:
                self.start_new_episode(self.train_env)

    def train(self, steps):
        self.populate_replay(self.initial_replay_size)

        best_score = 0.0
        steps_remaining = steps - self.initial_replay_size
        for step in range(steps_remaining):
            epsilon = self.epsilons[min(step, self.epsilon_decay_steps - 1)]
            action = self.choose_action(epsilon)
            step_result = self.train_env.step(action)
            frame, raw_reward, terminal, _ = step_result
            reward = self.process_reward(raw_reward)

            self.train_stats.add_reward(raw_reward)
            self.replay.insert(frame, action, reward, terminal)

            if terminal:
                self.start_new_episode(self.train_env)

            # train
            if step % self.training_freq == 0:
                sample = self.replay.sample()
                states, actions, rewards, next_states, terminals = sample

                loss = self.network.train(
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminals
                )
                self.train_stats.add_loss(loss)

            if step % self.summary_freq == 0:
                self.train_stats.summarize(step)

            # evaluate
            if step % self.eval_freq == 0:
                results = self.evaluate(
                    self.num_eval_episodes,
                    self.max_steps_per_eval_episode
                )
                print("Evaluation score at step {0}: {1}".format(step, results))

                self.eval_stats.summarize(step)
                if results > best_score:
                    self.network.save_checkpoint(step)
                    best_score = results

    def evaluate(self, num_episodes, max_steps_per_episode):
        """Evaluate the agent"""
        try:
            initial_state = self.eval_env.reset()
        except gym.error.Error:
            # We'll arrive here if we last exited eval by reaching the step
            # limit rather than finishing episodes. It's ugly, but just
            # continue where we left off.
            print('Continuing eval episode.')
            initial_state = self.eval_env.step(0)

        state = np.array([state[0] for state in initial_state])
        terminal = False

        total_reward = 0.0
        for episode in range(num_episodes):
            for _ in range(max_steps_per_episode):
                action = self.choose_eval_action(state, self.eval_epsilon)
                screen, reward, terminal, _ = self.eval_env.step(action)
                total_reward += reward

                state = np.append(
                    state[1:, :, :],
                    np.expand_dims(screen, 0),
                    axis=0
                )

                if self.eval_stats is not None:
                    self.eval_stats.add_reward(reward)

                if terminal:
                    try:
                        initial_state = self.eval_env.reset()
                        state = np.array(
                            [state[0] for state in initial_state]
                        )
                        break
                    except gym.error.Error:
                        # Just lost a life. Keep going.
                        pass

            episode += 1
            if self.eval_stats is not None:
                self.eval_stats.increment_episode_counter()

        return total_reward / num_episodes


class DoubleDeepQLearner(DeepQLearner):
    def create_network(self, config, num_actions, is_demo, run_name):
        return networks.DoubleQNetwork(config, num_actions, is_demo, run_name)
