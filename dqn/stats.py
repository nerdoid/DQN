import tensorflow as tf


class Stats:
    def __init__(self, config, run_name, is_eval):
        self.is_eval = is_eval
        self.reward = 0
        self.step_count = 0
        self.loss = 0.0
        self.loss_count = 0
        self.episodes = 0
        self.current_score = 0
        self.max_score = None
        self.min_score = None

        prefix = 'eval' if is_eval else 'train'

        self.path = 'results/stats/{0}/{1}/{2}/{3}'.format(
            config['game'],
            config['agent_type'].__name__,
            run_name,
            prefix
        )

        self.score_per_game = tf.placeholder(
            tf.float32,
            shape=[],
            name='score_per_episode'
        )
        self.total_steps_per_game = tf.placeholder(tf.float32, shape=[])
        self.max_reward = tf.placeholder(tf.float32, shape=[])
        self.min_reward = tf.placeholder(tf.float32, shape=[])

        with tf.name_scope('{}_summaries'.format(prefix)):
            self.score_per_game_summary = tf.summary.scalar(
                'score_per_episode',
                self.score_per_game
            )
            self.steps_per_game_summary = tf.summary.scalar(
                'steps_per_episode',
                self.total_steps_per_game
            )
            self.max_summary = tf.summary.scalar(
                'max_score',
                self.max_reward
            )
            self.min_summary = tf.summary.scalar(
                'min_score',
                self.min_reward
            )

            summary_ops = [
                self.score_per_game_summary,
                self.steps_per_game_summary,
                self.max_summary,
                self.min_summary
            ]

            if not is_eval:
                self.avg_loss = tf.placeholder(tf.float32, shape=[], name='loss')
                self.loss_summary = tf.summary.scalar('loss', self.avg_loss)
                summary_ops.append(self.loss_summary)

            self.summary_op = tf.summary.merge(summary_ops)

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(self.path)

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
