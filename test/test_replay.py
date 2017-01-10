"""Tests for replay memory"""
import numpy as np
import tensorflow as tf
from hedonist.replay import init_memory


class TestReplay:
    def setup_method(self):
        self.frame_pl = tf.placeholder(
            tf.uint8,
            shape=[1, 3, 4],
            name='observation'
        )
        self.action_pl = tf.placeholder(
            tf.int32,
            shape=[1],
            name='action'
        )
        self.reward_pl = tf.placeholder(
            tf.float32,
            shape=[1],
            name='reward'
        )
        self.terminal_pl = tf.placeholder(
            tf.bool,
            shape=[1],
            name='terminal'
        )

        self.sess = tf.Session()

    def test_add_same_frame_repeatedly(self):
        """Mimic the initial state wherein the first frame is duplicated
        history_length times to populate replay for first retrieved state
        """
        memory_ops = init_memory(
            capacity=10,
            screen_dims=(3, 4),
            state_size=4,
            batch_size=3,
            num_actions=6,
            frame_input=self.frame_pl,
            action_input=self.action_pl,
            reward_input=self.reward_pl,
            terminal_input=self.terminal_pl
        )
        insert_op, cur_state_op, _, _, _, _, _ = memory_ops

        self.sess.run(tf.global_variables_initializer())

        frame = np.array(
            [[
                [0,   63,  127, 191],
                [255, 191, 127, 63],
                [127, 191, 255, 0]
            ]]
        )
        action = [0]
        reward = [0.0]
        terminal = [False]

        for _ in range(4):
            self.sess.run(
                insert_op,
                feed_dict={
                    self.frame_pl: frame,
                    self.action_pl: action,
                    self.reward_pl: reward,
                    self.terminal_pl: terminal
                }
            )

        expected = np.array(
            [
                [
                    [0,   0,   0,   0],
                    [63,  63,  63,  63],
                    [127, 127, 127, 127],
                    [191, 191, 191, 191]
                ],
                [
                    [255, 255, 255, 255],
                    [191, 191, 191, 191],
                    [127, 127, 127, 127],
                    [63,  63,  63,  63]
                ],
                [
                    [127, 127, 127, 127],
                    [191, 191, 191, 191],
                    [255, 255, 255, 255],
                    [0,   0,   0,   0]
                ]
            ]
        )

        result = self.sess.run(cur_state_op)
        assert np.array_equal(result, expected)


    def test_add_unique_frames(self):
        """Mimic a state that is populated with unique consecutive frames"""
        memory_ops = init_memory(
            capacity=10,
            screen_dims=(3, 4),
            state_size=4,
            batch_size=3,
            num_actions=6,
            frame_input=self.frame_pl,
            action_input=self.action_pl,
            reward_input=self.reward_pl,
            terminal_input=self.terminal_pl
        )
        insert_op, cur_state_op, _, _, _, _, _ = memory_ops

        self.sess.run(tf.global_variables_initializer())

        frame_1 = np.array(
            [[
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34]
            ]]
        )
        frame_2 = np.array(
            [[
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64]
            ]]
        )
        frame_3 = np.array(
            [[
                [71, 72, 73, 74],
                [81, 82, 83, 84],
                [91, 92, 93, 94]
            ]]
        )
        frame_4 = np.array(
            [[
                [101, 102, 103, 104],
                [111, 112, 113, 114],
                [121, 122, 123, 124]
            ]]
        )

        action = [0]
        reward = [0.0]
        terminal = [False]

        for frame in [frame_1, frame_2, frame_3, frame_4]:
            self.sess.run(
                insert_op,
                feed_dict={
                    self.frame_pl: frame,
                    self.action_pl: action,
                    self.reward_pl: reward,
                    self.terminal_pl: terminal
                }
            )

        expected = np.array(
            [
                [
                    [11, 41, 71, 101],
                    [12, 42, 72, 102],
                    [13, 43, 73, 103],
                    [14, 44, 74, 104]
                ],
                [
                    [21, 51, 81, 111],
                    [22, 52, 82, 112],
                    [23, 53, 83, 113],
                    [24, 54, 84, 114]
                ],
                [
                    [31, 61, 91, 121],
                    [32, 62, 92, 122],
                    [33, 63, 93, 123],
                    [34, 64, 94, 124]
                ]
            ]
        )

        result = self.sess.run(cur_state_op)
        assert np.array_equal(result, expected)


    def add_test_frames(self, insert_op, actions, rewards, terminals):
        frame_1 = np.array([
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34]
            ]
        ])
        frame_2 = np.array([
            [
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64]
            ]
        ])
        frame_3 = np.array([
            [
                [71, 72, 73, 74],
                [81, 82, 83, 84],
                [91, 92, 93, 94]
            ]
        ])
        frame_4 = np.array([
            [
                [101, 102, 103, 104],
                [111, 112, 113, 114],
                [121, 122, 123, 124]
            ]
        ])
        advance_frame = np.array([
            [
                [127, 127, 127, 127],
                [191, 191, 191, 191],
                [255, 255, 255, 255]
            ]
        ])

        frames = [frame_1, frame_2, frame_3, frame_4, advance_frame]

        for i in range(5):
            self.sess.run(
                insert_op,
                feed_dict={
                    self.frame_pl: frames[i],
                    self.action_pl: actions[i],
                    self.reward_pl: rewards[i],
                    self.terminal_pl: terminals[i]
                }
            )


    def test_advance_state_window(self):
        """Does it return the most recent history_length frames, ignoring
        frames outside of the window?
        """
        memory_ops = init_memory(
            capacity=10,
            screen_dims=(3, 4),
            state_size=4,
            batch_size=3,
            num_actions=6,
            frame_input=self.frame_pl,
            action_input=self.action_pl,
            reward_input=self.reward_pl,
            terminal_input=self.terminal_pl
        )
        insert_op, cur_state_op, _, _, _, _, _ = memory_ops

        self.sess.run(tf.global_variables_initializer())

        actions = [[0]] * 5
        rewards = [[0.0]] * 5
        terminals = [[False]] * 5

        self.add_test_frames(insert_op, actions, rewards, terminals)

        expected = np.array(
            [
                [
                    [41, 71, 101, 127],
                    [42, 72, 102, 127],
                    [43, 73, 103, 127],
                    [44, 74, 104, 127]
                ],
                [
                    [51, 81, 111, 191],
                    [52, 82, 112, 191],
                    [53, 83, 113, 191],
                    [54, 84, 114, 191]
                ],
                [
                    [61, 91, 121, 255],
                    [62, 92, 122, 255],
                    [63, 93, 123, 255],
                    [64, 94, 124, 255]
                ]
            ]
        )

        result = self.sess.run(cur_state_op)
        assert np.array_equal(result, expected)


    def test_wrap(self):
        """Mimic adding past the capacity of the replay memory"""
        memory_ops = init_memory(
            capacity=4,
            screen_dims=(3, 4),
            state_size=4,
            batch_size=3,
            num_actions=6,
            frame_input=self.frame_pl,
            action_input=self.action_pl,
            reward_input=self.reward_pl,
            terminal_input=self.terminal_pl
        )
        insert_op, cur_state_op, _, _, _, _, _ = memory_ops

        self.sess.run(tf.global_variables_initializer())

        frame_1 = np.array([
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34]
            ]
        ])
        frame_2 = np.array([
            [
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64]
            ]
        ])
        frame_3 = np.array([
            [
                [71, 72, 73, 74],
                [81, 82, 83, 84],
                [91, 92, 93, 94]
            ]
        ])
        frame_4 = np.array([
            [
                [101, 102, 103, 104],
                [111, 112, 113, 114],
                [121, 122, 123, 124]
            ]
        ])
        wrap_frame = np.array([
            [
                [127, 127, 127, 127],
                [191, 191, 191, 191],
                [255, 255, 255, 255]
            ]
        ])

        frames = [frame_1, frame_2, frame_3, frame_4, wrap_frame]

        action = [0]
        reward = [0.0]
        terminal = [False]

        for frame in frames:
            self.sess.run(
                insert_op,
                feed_dict={
                    self.frame_pl: frame,
                    self.action_pl: action,
                    self.reward_pl: reward,
                    self.terminal_pl: terminal
                }
            )

        expected = np.array(
            [
                [
                    [41, 71, 101, 127],
                    [42, 72, 102, 127],
                    [43, 73, 103, 127],
                    [44, 74, 104, 127]
                ],
                [
                    [51, 81, 111, 191],
                    [52, 82, 112, 191],
                    [53, 83, 113, 191],
                    [54, 84, 114, 191]
                ],
                [
                    [61, 91, 121, 255],
                    [62, 92, 122, 255],
                    [63, 93, 123, 255],
                    [64, 94, 124, 255]
                ]
            ]
        )

        result = self.sess.run(cur_state_op)
        assert np.array_equal(result, expected)


    def test_single_sample(self):
        """Verify a single sample"""
        memory_ops = init_memory(
            capacity=10,
            screen_dims=(3, 4),
            state_size=4,
            batch_size=1,
            num_actions=6,
            frame_input=self.frame_pl,
            action_input=self.action_pl,
            reward_input=self.reward_pl,
            terminal_input=self.terminal_pl
        )
        insert_op, cur_state_op, state_sample, action_sample, reward_sample, next_state_sample, terminal_sample = memory_ops

        self.sess.run(tf.global_variables_initializer())

        actions = [[0], [1], [2], [3], [4]]
        rewards = [[0.0], [-1.0], [2.0], [0.0], [3.0]]
        terminals = [[False], [False], [False], [False], [True]]

        self.add_test_frames(insert_op, actions, rewards, terminals)

        sample = self.sess.run(
            [
                state_sample,
                action_sample,
                reward_sample,
                next_state_sample,
                terminal_sample
            ]
        )

        expected_first_state = np.array([
            [
                [
                    [11, 41, 71, 101],
                    [12, 42, 72, 102],
                    [13, 43, 73, 103],
                    [14, 44, 74, 104]
                ],
                [
                    [21, 51, 81, 111],
                    [22, 52, 82, 112],
                    [23, 53, 83, 113],
                    [24, 54, 84, 114]
                ],
                [
                    [31, 61, 91, 121],
                    [32, 62, 92, 122],
                    [33, 63, 93, 123],
                    [34, 64, 94, 124]
                ]
            ]
        ])

        expected_next_state = np.array([
            [
                [
                    [41, 71, 101, 127],
                    [42, 72, 102, 127],
                    [43, 73, 103, 127],
                    [44, 74, 104, 127]
                ],
                [
                    [51, 81, 111, 191],
                    [52, 82, 112, 191],
                    [53, 83, 113, 191],
                    [54, 84, 114, 191]
                ],
                [
                    [61, 91, 121, 255],
                    [62, 92, 122, 255],
                    [63, 93, 123, 255],
                    [64, 94, 124, 255]
                ]
            ]
        ])

        assert len(sample) == 5
        assert np.array_equal(sample[0], expected_first_state)
        assert np.array_equal(sample[1], np.array([[0, 0, 0, 0, 1, 0]]))
        assert np.array_equal(sample[2], np.array([3]))
        assert np.array_equal(sample[3], expected_next_state)
        assert np.array_equal(sample[4], np.array([True]))
