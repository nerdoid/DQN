"""Tests for replay memory"""
import numpy as np
import tensorflow as tf
from hedonist.replay import Replay


class TestReplay:
    def setup_method(self):
        self.sess = tf.Session()

    def test_add_same_frame_repeatedly(self):
        """Mimic the initial state wherein the first frame is duplicated
        history_length times to populate replay for first retrieved state
        """
        self.replay = Replay(self.sess, 10, (3, 4), 4, 3, 6)
        self.sess.run(tf.global_variables_initializer())

        frame = np.array(
            [
                [0,   63,  127, 191],
                [255, 191, 127, 63],
                [127, 191, 255, 0]
            ]
        )
        action = 0
        reward = 0.0
        terminal = False

        for _ in range(4):
            self.replay.insert(frame, action, reward, terminal)

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

        result = self.replay.get_current_state()
        assert np.array_equal(result, expected)


    def test_add_unique_frames(self):
        """Mimic a state that is populated with unique consecutive frames"""
        self.replay = Replay(self.sess, 10, (3, 4), 4, 3, 6)
        self.sess.run(tf.global_variables_initializer())

        frame_1 = np.array(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34]
            ]
        )
        frame_2 = np.array(
            [
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64]
            ]
        )
        frame_3 = np.array(
            [
                [71, 72, 73, 74],
                [81, 82, 83, 84],
                [91, 92, 93, 94]
            ]
        )
        frame_4 = np.array(
            [
                [101, 102, 103, 104],
                [111, 112, 113, 114],
                [121, 122, 123, 124]
            ]
        )

        action = 0
        reward = 0.0
        terminal = False

        for frame in [frame_1, frame_2, frame_3, frame_4]:
            self.replay.insert(frame, action, reward, terminal)

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

        result = self.replay.get_current_state()
        assert np.array_equal(result, expected)


    def add_test_frames(self, actions, rewards, terminals):
        frame_1 = np.array(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34]
            ]
        )
        frame_2 = np.array(
            [
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64]
            ]
        )
        frame_3 = np.array(
            [
                [71, 72, 73, 74],
                [81, 82, 83, 84],
                [91, 92, 93, 94]
            ]
        )
        frame_4 = np.array(
            [
                [101, 102, 103, 104],
                [111, 112, 113, 114],
                [121, 122, 123, 124]
            ]
        )
        advance_frame = np.array(
            [
                [127, 127, 127, 127],
                [191, 191, 191, 191],
                [255, 255, 255, 255]
            ]
        )

        frames = [frame_1, frame_2, frame_3, frame_4, advance_frame]

        for i in range(5):
            self.replay.insert(frames[i], actions[i], rewards[i], terminals[i])


    def test_advance_state_window(self):
        """Does it return the most recent history_length frames, ignoring
        frames outside of the window?
        """
        self.replay = Replay(self.sess, 10, (3, 4), 4, 3, 6)
        self.sess.run(tf.global_variables_initializer())

        actions = [0] * 5
        rewards = [0.0] * 5
        terminals = [False] * 5

        self.add_test_frames(actions, rewards, terminals)

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

        result = self.replay.get_current_state()
        assert np.array_equal(result, expected)


    def test_wrap(self):
        """Mimic adding past the capacity of the replay memory"""
        self.replay = Replay(self.sess, 4, (3, 4), 4, 3, 6)
        self.sess.run(tf.global_variables_initializer())

        frame_1 = np.array(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34]
            ]
        )
        frame_2 = np.array(
            [
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64]
            ]
        )
        frame_3 = np.array(
            [
                [71, 72, 73, 74],
                [81, 82, 83, 84],
                [91, 92, 93, 94]
            ]
        )
        frame_4 = np.array(
            [
                [101, 102, 103, 104],
                [111, 112, 113, 114],
                [121, 122, 123, 124]
            ]
        )
        wrap_frame = np.array(
            [
                [127, 127, 127, 127],
                [191, 191, 191, 191],
                [255, 255, 255, 255]
            ]
        )

        frames = [frame_1, frame_2, frame_3, frame_4, wrap_frame]

        action = 0
        reward = 0.0
        terminal = False

        for frame in frames:
            self.replay.insert(frame, action, reward, terminal)

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

        result = self.replay.get_current_state()
        assert np.array_equal(result, expected)


    def test_single_sample(self):
        """Verify a single sample"""
        self.replay = Replay(self.sess, 10, (3, 4), 4, 1, 6)
        self.sess.run(tf.global_variables_initializer())

        actions = [0, 1, 2, 3, 4]
        rewards = [0.0, -1.0, 2.0, 0.0, 3.0]
        terminals = [False, False, False, False, True]

        self.add_test_frames(actions, rewards, terminals)

        sample = self.replay.sample()

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

    def test_no_terminals_in_first_state_1(self):
        """Verify that the sample is not the one with a terminal in the first
        state.
        """
        self.replay = Replay(self.sess, 10, (3, 4), 4, 1, 6)
        self.sess.run(tf.global_variables_initializer())

        actions = [0, 1, 2, 3, 4]
        rewards = [0.0, -1.0, 2.0, 0.0, 3.0]
        terminals = [False, False, False, False, True]

        self.add_test_frames(actions, rewards, terminals)

        frame = np.array(
            [
                [95, 95, 95, 95],
                [159, 159, 159, 159],
                [221, 221, 221, 221]
            ]
        )

        self.replay.insert(frame, 3, 1.0, False)

        sample = self.replay.sample()

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


    def test_no_terminals_in_first_state_2(self):
        """Verify that the sample is not the one with a terminal in the first
        state.
        """
        self.replay = Replay(self.sess, 10, (3, 4), 4, 1, 6)
        self.sess.run(tf.global_variables_initializer())

        actions = [0, 1, 2, 3, 4]
        rewards = [0.0, -1.0, 2.0, 0.0, 3.0]
        terminals = [True, False, False, False, False]

        self.add_test_frames(actions, rewards, terminals)

        frame = np.array(
            [
                [95, 95, 95, 95],
                [159, 159, 159, 159],
                [221, 221, 221, 221]
            ]
        )

        self.replay.insert(frame, 3, 1.0, True)

        sample = self.replay.sample()

        expected_first_state = np.array([
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

        expected_next_state = np.array([
            [
                [
                    [71, 101, 127, 95],
                    [72, 102, 127, 95],
                    [73, 103, 127, 95],
                    [74, 104, 127, 95]
                ],
                [
                    [81, 111, 191, 159],
                    [82, 112, 191, 159],
                    [83, 113, 191, 159],
                    [84, 114, 191, 159]
                ],
                [
                    [91, 121, 255, 221],
                    [92, 122, 255, 221],
                    [93, 123, 255, 221],
                    [94, 124, 255, 221]
                ]
            ]
        ])

        assert len(sample) == 5
        assert np.array_equal(sample[0], expected_first_state)
        assert np.array_equal(sample[1], np.array([[0, 0, 0, 1, 0, 0]]))
        assert np.array_equal(sample[2], np.array([1.0]))
        assert np.array_equal(sample[3], expected_next_state)
        assert np.array_equal(sample[4], np.array([True]))


    def test_sample_alignment(self):
        """Verify that sample data lines up"""
        self.replay = Replay(self.sess, 10, (3, 4), 4, 32, 6)
        self.sess.run(tf.global_variables_initializer())

        actions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        rewards = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0]
        terminals = [
            False, False, False, True, False, False, False, False, False, False
        ]

        frame = np.array(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]
            ]
        )

        for i in range(len(actions)):
            self.replay.insert(frame * i, actions[i], rewards[i], terminals[i])

        batch = self.replay.sample()
        b_frames1, b_actions, b_rewards, b_frames2, b_terminals = batch

        for sample_i, r in enumerate(b_rewards):
            input_i = rewards.index(r)
            b_action = np.nonzero(b_actions[sample_i])[0][0]
            assert b_action == actions[input_i]
            assert b_terminals[sample_i] == terminals[input_i]
