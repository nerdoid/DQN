import gym
from gym import wrappers
import tensorflow as tf
import numpy as np


class Atari:
    """Wrapper around OpenAI gym atari environments."""
    def __init__(self, config, monitor=False, monitor_freq=50,
                 monitor_name=None, run_name=None):
        self.screen_width, self.screen_height = config['screen_dims']
        self.history_length = config['history_length']
        self.monitor = monitor

        self.env = gym.make(config['game'])

        if monitor:
            assert monitor_name is not None, 'monitor_name is blank'

            monitor_path = 'results/videos/{0}/{1}'.format(
                config['game'],
                config['agent_type'].__name__
            )

            if run_name:
                monitor_path = monitor_path + '/' + run_name

            monitor_path = monitor_path + '/' + monitor_name

            self.env = wrappers.Monitor(
                self.env,
                monitor_path,
                force=True,
                video_callable=lambda count: count % monitor_freq == 0
            )
            self.ale = self.env.env.env.ale # OK then...
        else:
            self.ale = self.env.ale

        # Frame processor
        self.frame_placeholder = tf.placeholder(
            shape=[210, 160, 3],
            dtype=tf.uint8
        )
        self.grayscale_op = tf.image.rgb_to_grayscale(self.frame_placeholder)
        self.resize_op = tf.image.resize_images(
            self.grayscale_op,
            [self.screen_width, self.screen_height]
        )
        self.sess = tf.Session()

        self.num_actions = len(self.ale.getMinimalActionSet())
        self.possible_actions = range(self.num_actions)

        self.reset()

        print(
            'repeat action prob: {}'.format(
                self.ale.getFloat(b'repeat_action_probability')
            )
        )
        print('possible actions: {}'.format(self.possible_actions))

    def __getattr__(self, name):
        """Fallback to gym env if we don't define it here."""
        return getattr(self.env, name)

    def get_possible_actions(self):
        """Return list of possible actions for game"""
        return self.possible_actions

    def reset(self):
        """Reset the env and return initial state"""
        frame = self.env.reset()
        state = [(self.preprocess(frame), 0, 0, False)]
        state = state * self.history_length
        return state

    def step(self, action):
        """Apply action to game and return next screen and reward"""
        lives_before = self.ale.lives()
        frame, reward, done, info = self.env.step(action)
        lives_after = self.ale.lives()

        # End the episode when a life is lost
        if lives_before > lives_after:
            done = True

        return (self.preprocess(frame), reward, done, info)

    def preprocess(self, frame):
        """Preprocess frame for agent"""
        processed = self.sess.run(
            self.resize_op,
            {self.frame_placeholder: frame}
        )

        return np.squeeze(processed)
