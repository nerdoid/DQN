import tensorflow as tf


class Stats:
    def __init__(self, config, is_eval):
        self.is_eval = is_eval
        self.reward = 0
        self.step_count = 0
        self.loss = 0.0
        self.loss_count = 0
        self.episodes = 0
        self.current_score = 0
        self.max_score = None
        self.min_score = None

        self.score_per_game = tf.placeholder(
            tf.float32,
            shape=[],
            name='score_per_episode'
        )
        self.total_steps_per_game = tf.placeholder(tf.float32, shape=[])
        self.max_reward = tf.placeholder(tf.float32, shape=[])
        self.min_reward = tf.placeholder(tf.float32, shape=[])

        self.score_per_game_summary = tf.scalar_summary(
            'Score Per Episode',
            self.score_per_game
        )
        self.steps_per_game_summary = tf.scalar_summary(
            'Steps Per Episode',
            self.total_steps_per_game
        )
        self.max_summary = tf.scalar_summary('Max Score', self.max_reward)
        self.min_summary = tf.scalar_summary('Min Score', self.min_reward)

        self.path = 'results/stats/{0}/{1}'.format(
            config['game'],
            config['agent_type'].__name__
        )

        if not is_eval:
            self.avg_loss = tf.placeholder(tf.float32, shape=[], name='loss')
            self.loss_summary = tf.scalar_summary('Loss', self.avg_loss)
            self.summary_op = tf.merge_summary(
                [
                    self.score_per_game_summary,
                    self.steps_per_game_summary,
                    self.loss_summary,
                    self.max_summary,
                    self.min_summary
                ]
            )
            self.path = self.path + '/train'
        else:
            self.summary_op = tf.merge_summary(
                [
                    self.score_per_game_summary,
                    self.steps_per_game_summary,
                    self.max_summary,
                    self.min_summary
                ]
            )
            self.path = self.path + '/eval'

        self.sess = tf.Session()
        self.summary_writer = tf.train.SummaryWriter(self.path)

    def summarize(self, step):
        avg_loss = 0
        if self.loss_count != 0:
            avg_loss = self.loss / self.loss_count

        score_per_episode = 0.0
        steps_per_episode = 0

        if self.episodes == 0:
            score_per_episode = self.reward
            steps_per_episode = self.step_count
        else:
            score_per_episode = self.reward / self.episodes
            steps_per_episode = self.step_count / self.episodes

        score_per_episode = float(score_per_episode)

        if not self.is_eval:
            summary_str = self.sess.run(
                self.summary_op,
                feed_dict={
                    self.score_per_game: score_per_episode,
                    self.avg_loss: avg_loss,
                    self.total_steps_per_game: steps_per_episode,
                    self.max_reward: self.max_score if self.max_score else 0.0,
                    self.min_reward: self.min_score if self.min_score else 0.0
                }
            )
            self.summary_writer.add_summary(summary_str, global_step=step)
        else:
            summary_str = self.sess.run(
                self.summary_op,
                feed_dict={
                    self.score_per_game: score_per_episode,
                    self.total_steps_per_game: steps_per_episode,
                    self.max_reward: self.max_score if self.max_score else 0.0,
                    self.min_reward: self.min_score if self.min_score else 0.0
                }
            )
            self.summary_writer.add_summary(summary_str, global_step=step)

        self.reward = 0
        self.step_count = 0
        self.loss = 0
        self.loss_count = 0
        self.episodes = 0
        self.max_score = None
        self.min_score = None

    def add_reward(self, reward):
        self.reward += reward
        self.current_score += reward

        self.step_count += 1

    def add_loss(self, loss):
        self.loss += loss
        self.loss_count += 1

    def increment_episode_counter(self):
        self.episodes += 1

        if self.max_score is None or self.current_score > self.max_score:
            self.max_score = self.current_score
        if self.min_score is None or self.current_score < self.min_score:
            self.min_score = self.current_score

        self.current_score = 0
