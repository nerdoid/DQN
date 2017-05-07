"""Defines replay memory"""
import tensorflow as tf
import numpy as np


def modulo(numbers, capacity):
    """Provides positive AND negative module since TensorFlow does not support
    the negative modulo. This behavior is useful for sampling states that wrap
    around the replay memory.

    Returns: The wrapped numbers in range [0, capacity]

    Args:
        numbers: A [None] tensor of scalars
        capacity: Max size of the replay memory
    """
    return tf.add(numbers, capacity) % capacity


class Replay():
    def __init__(self, sess, capacity, screen_dims, state_size, batch_size,
                 num_actions):
        self.sess = sess
        self.capacity = capacity
        self.state_size = state_size
        self.batch_size = batch_size

        self.frame_input = tf.placeholder(
            tf.uint8,
            shape=[1, screen_dims[0], screen_dims[1]],
            name='observation'
        )
        self.action_input = tf.placeholder(
            tf.int32,
            shape=[1],
            name='action'
        )
        self.reward_input = tf.placeholder(
            tf.float32,
            shape=[1],
            name='reward'
        )
        self.terminal_input = tf.placeholder(
            tf.bool,
            shape=[1],
            name='terminal'
        )

        self.frames = tf.Variable(
            tf.zeros([capacity, screen_dims[0], screen_dims[1]], dtype=tf.uint8)
        )
        self.actions = tf.Variable(
            tf.zeros([capacity], dtype=tf.int32)
        )
        self.rewards = tf.Variable(
            tf.zeros([capacity], dtype=tf.float32)
        )
        self.terminals = tf.Variable(
            tf.zeros([capacity], dtype=tf.bool)
        )

        self.index = tf.Variable(0, name='replay_index')
        self.size = tf.Variable(0, name='replay_size')

        self.batch_indices = self.get_batch_indices()
        self.batch_slices = self.get_batch_slices()

        self.insert_frame = self.build_insert_op(
            self.frames, self.frame_input
        )
        self.insert_action = self.build_insert_op(
            self.actions, self.action_input
        )
        self.insert_reward = self.build_insert_op(
            self.rewards, self.reward_input
        )
        self.insert_terminal = self.build_insert_op(
            self.terminals, self.terminal_input
        )

        insert_ops = [
            self.insert_frame,
            self.insert_action,
            self.insert_reward,
            self.insert_terminal
        ]
        # Ensure that insertions occur before index and size are updated.
        # Without the control dependency, the states of index and size at time
        # of insertion would be nondeterministic.
        with tf.control_dependencies(insert_ops):
            index_update = self.build_index_update()
            size_update = self.build_size_update()

        self.insert_op = tf.group(
            self.insert_frame,
            self.insert_action,
            self.insert_reward,
            self.insert_terminal,
            index_update,
            size_update
        )

        state_samples = self.build_sample_frames_op(self.frames)
        self.sample_states, self.sample_next_states = state_samples
        actions_sample = self.build_sample_op(self.actions)
        self.sample_actions = tf.one_hot(actions_sample, num_actions)
        self.sample_rewards = self.build_sample_op(self.rewards)
        self.sample_terminals = self.build_sample_op(self.terminals)

        self.current_state = self.build_current_state_op()

    def build_insert_op(self, buffer, buffer_input):
        """
        Args:
            buffer_input: [1] tensor

        Returns:
            Op to add new memory to replay memory
        """
        return tf.scatter_update(
            buffer,
            tf.reshape(self.index, [1]),
            buffer_input
        )

    def build_sample_op(self, buffer):
        """
        Args:
            buffer: [capacity] tensor with a type of memory (action, terminal,
                    etc)
        Returns:
            Op to sample a mini-batch from replay memory
        """
        return tf.gather(
            buffer,
            self.batch_indices
        )

    def build_sample_frames_op(self, buffer):
        """Samples the frame replay memory to produce the state and the next-
        state sample.

        Args:
            buffer: [capacity] tensor representing the frame history.

        Returns:
            Ops to sample a mini-batch for the state and next state.
        """
        next_state_sample = tf.transpose(
            tf.gather(
                self.frames,
                self.batch_slices
            ),
            perm=[0, 2, 3, 1]
        )

        state_sample = tf.transpose(
            tf.gather(
                self.frames,
                modulo(self.batch_slices - 1, self.capacity)
            ),
            perm=[0, 2, 3, 1]
        )

        return state_sample, next_state_sample

    def get_batch_indices(self):
        """Generates the replay memory indices to be used for the mini-batch.

        Returns:
            An op that produces indices for the mini-batch.
        """
        # These min and max values ensure that we only allow wrapped samples
        # if the memory is at capacity. Otherwise the first and last states
        # are not actually correlated and should not be sampled together.
        replay_size_plus_one = self.size + 1
        min_value = tf.cond(
            tf.equal(replay_size_plus_one, self.capacity),
            lambda: tf.constant(0),
            lambda: tf.constant(self.state_size)
        )
        max_value = tf.cond(
            tf.equal(replay_size_plus_one, self.capacity),
            lambda: tf.constant(self.capacity),
            lambda: self.size
        )

        def attempt_sample(b):
            """Tries to pick a random index and add it to the batch. Fails if
            the index would mean the first state (consisting of N frames) would
            represent a terminal state. These samples would mislead the agent
            because they occured in different episodes and are unrelated.

            Args:
                b: A tensor list of indices sampled so far.
            Returns:
                A (possibly) updated tensor list of indices.
            """
            sample_index = tf.random_uniform(
                [1], min_value, max_value, dtype=tf.int32
            )
            sample_slice = tf.map_fn(
                lambda i: modulo(tf.range(i - self.state_size, i), self.capacity),
                sample_index
            )
            prev_terminals = tf.gather(self.terminals, sample_slice)
            terminal_sum = tf.reduce_sum(tf.cast(prev_terminals, tf.int32))

            return tf.cond(
                tf.equal(terminal_sum, tf.constant(0)),
                lambda: tf.concat([b, sample_index], 0),
                lambda: b
            )

        batch_indices = tf.zeros([0], dtype=tf.int32)
        condition = lambda b: tf.less(tf.size(b), self.batch_size)
        body = attempt_sample
        return tf.while_loop(
            condition,
            body,
            [batch_indices],
            shape_invariants=[tf.TensorShape([None])]
        )

    def get_batch_slices(self):
        """Converts sample indices to slices for sampling the frames to produce
        states of N frames.

        Returns:
            An op to yield frame slices.
        """
        # States are N consecutive frames. So we need to convert each index
        # into a vector of the form [index - N + 1, index - N + 2, ..., index]
        batch_slices = tf.map_fn(
            lambda i: modulo(tf.range(i - self.state_size + 1, i + 1), self.capacity),
            self.batch_indices
        )

        return batch_slices

    def build_current_state_op(self):
        """Builds op to retrieve the state from the current index.

        Returns:
            A [state_size, 84, 84] tensor representing the state
        """
        return tf.transpose(
            tf.gather(
                self.frames,
                modulo(
                    tf.range(self.index - self.state_size, self.index),
                    self.capacity
                )
            ),
            perm=[1, 2, 0]
        )

    def build_index_update(self):
        """Builds an op that increments the index and wrap it upon reaching
        the capacity.

        Returns:
            An op that increments the index
        """
        index_plus_one = self.index + 1
        index_update = tf.assign(
            self.index,
            index_plus_one % tf.constant(self.capacity),
            name='replay_index_update'
        )

        return index_update

    def build_size_update(self):
        """Builds an op that increments the current size of the replay memory
        until it reaches capacity. Once reached, the size then stays at
        capacity.

        Returns:
            An op that increments the size
        """
        size_plus_one = self.size + 1
        next_size = tf.cond(
            tf.equal(size_plus_one, self.capacity),
            lambda: self.size,
            lambda: size_plus_one
        )
        size_update = tf.assign(
            self.size, next_size, name='replay_size_update'
        )

        return size_update

    def insert(self, frame, action, reward, terminal):
        """Insert a new memory into the replay memory.

        Args:
            frame: A [screen_size, screen_size] numpy array
            action: A scalar
            reward: A float32
            terminal: A boolean
        """
        self.sess.run(
            self.insert_op,
            feed_dict={
                self.frame_input: [frame],
                self.action_input: [action],
                self.reward_input: [reward],
                self.terminal_input: [terminal]
            }
        )

    def get_current_state(self):
        """Gets the last N frames"""
        return self.sess.run(self.current_state)

    def sample(self):
        """Gets a mini-batch from the replay memory.

        Returns:
            A list of lists where each list is the batch for the corresponding
            part of the memory (states, actions, rewards, next_states,
            terminals).
        """
        return self.sess.run(
            [
                self.sample_states,
                self.sample_actions,
                self.sample_rewards,
                self.sample_next_states,
                self.sample_terminals
            ]
        )
