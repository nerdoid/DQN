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


def init_rewards(capacity, index, reward_input, batch_indices, name):
    """Builds ops to insert into and sample from the reward replay memory.

    Args:
        capacity: Max size of the replay memory
        index: Current index for this time step. Determines point of insertion.
        reward_input: A tensor with a single tf.float32
        batch_indices: A tensor containing scalars or slices for batch sampling

    Returns:
        insert_reward: Op to add reward to replay memory
        reward_sample: Op to sample a mini-batch from replay memory
    """
    rewards = tf.Variable(
        tf.zeros([capacity], dtype=tf.float32)
    )

    insert_reward = tf.scatter_update(
        rewards,
        tf.reshape(index, [1]),
        reward_input
    )

    reward_sample = tf.gather(
        rewards,
        batch_indices
    )

    return insert_reward, reward_sample


def init_terminals(capacity, index, terminal_input, batch_indices, name):
    """Builds ops to insert into and sample from the terminal replay memory.

    Args:
        capacity: Max size of the replay memory
        index: Current index for this time step. Determines point of insertion.
        terminal_input: A tensor with a single boolean
        batch_indices: A tensor containing scalars or slices for batch sampling

    Returns:
        insert_terminal: Op to add terminal flag to replay memory
        terminal_sample: Op to sample a mini-batch from replay memory
    """
    terminals = tf.Variable(
        tf.zeros([capacity], dtype=tf.bool)
    )

    insert_terminal = tf.scatter_update(
        terminals,
        tf.reshape(index, [1]),
        terminal_input
    )

    terminal_sample = tf.gather(
        terminals,
        batch_indices
    )

    return insert_terminal, terminal_sample


def init_actions(capacity, index, action_input, batch_indices, num_actions,
                 name):

    """Builds ops to insert into and sample from the action replay memory.

    Args:
        capacity: Max size of the replay memory
        index: Current index for this time step. Determines point of insertion.
        action_input: A tensor with a single tf.int32
        batch_indices: A tensor containing scalars or slices for batch sampling

    Returns:
        insert_action: Op to add an action to replay memory
        one_hot_sample: Op to sample a mini-batch of one-hot actions from
                        replay memory
    """
    actions = tf.Variable(
        tf.zeros([capacity], dtype=tf.int32)
    )

    insert_action = tf.scatter_update(
        actions,
        tf.reshape(index, [1]),
        action_input
    )

    action_sample = tf.gather(
        actions,
        batch_indices
    )

    one_hot_sample = tf.one_hot(action_sample, num_actions)

    return insert_action, one_hot_sample


def get_current_state(capacity, index, state_size, frames):
    """Builds op to retrieve the state from the current index.

    Args:
        capacity: Max size of the replay memory
        index: Current index for this time step. Determines point of insertion.
        state_size: A [1] tensor representing # of frames in a state
        frames: A [capacity, 84, 84] tensor for the frame replay memory

    Returns:
        A [state_size, 84, 84] tensor representing the state
    """
    return tf.transpose(
        tf.gather(
            frames,
            modulo(tf.range(index - state_size, index), capacity)
        ),
        perm=[1, 2, 0]
    )


def init_frames(capacity, screen_dims, state_size, index, frame_input,
                batch_slices, name=None):
    """Builds ops to insert into and sample from the frame replay memory.

    Args:
        capacity: Max size of the replay memory
        screen_dims: A (height, width) tuple
        state_size: A [1] tensor representing # of frames in a state
        index: Current index for this time step. Determines point of insertion.
        frame_input: A [1, 84, 84] tensor of type tf.uint8
        batch_indices: A tensor containing scalars or slices for batch sampling

    Returns:
        insert_action: Op to add an action to replay memory
        current_state: Op that retrieves the state from the current index
        state_sample: Op to sample a mini-batch of states from replay memory
        next_state_sample: Op to sample a mini-batch of next states from replay
                           memory
    """
    frames = tf.Variable(
        tf.zeros([capacity, screen_dims[0], screen_dims[1]], dtype=tf.uint8)
    )

    insert_frame = tf.scatter_update(
        frames,
        tf.reshape(index, [1]),
        frame_input
    )

    next_state_sample = tf.transpose(
        tf.gather(
            frames,
            batch_slices
        ),
        perm=[0, 2, 3, 1]
    )

    state_sample = tf.transpose(
        tf.gather(
            frames,
            modulo(batch_slices - 1, capacity)
        ),
        perm=[0, 2, 3, 1]
    )

    current_state = get_current_state(capacity, index, state_size, frames)

    return insert_frame, current_state, state_sample, next_state_sample


def init_index_update(capacity, index):
    """Builds an op that increments the index and will wrap upon reaching the
    capacity.

    Args:
        capacity: Max size of the replay memory
        index: Current index for this time step. Determines point of insertion.

    Returns:
        An op that increments the index
    """
    index_plus_one = index + 1
    index_update = tf.assign(
        index,
        index_plus_one % tf.constant(capacity),
        name='replay_index_update'
    )

    return index_update


def init_size_update(capacity, size):
    """Builds an op that increments the current size of the replay memory until
    it reaches capacity. The size then stays at the capacity.

    Args:
        capacity: Max size of the replay memory
        index: Current index for this time step. Determines point of insertion.

    Returns:
        An op that increments the size
    """

    # Size - will hold at capacity
    size_plus_one = size + 1
    next_size = tf.cond(
        tf.equal(size_plus_one, capacity),
        lambda: size,
        lambda: size_plus_one
    )
    size_update = tf.assign(size, next_size, name='replay_size_update')

    return size_update


def get_batch_slices(batch_size, state_size, replay_size, capacity):
    # These min and max values ensure that we only allow wrapped samples if the
    # memory is at capacity. Otherwise the first and last states are not
    # actually correlated and should not be sampled together.
    replay_size_plus_one = replay_size + 1
    min_value = tf.cond(
        tf.equal(replay_size_plus_one, capacity),
        lambda: tf.constant(0),
        lambda: tf.constant(state_size)
    )
    max_value = tf.cond(
        tf.equal(replay_size_plus_one, capacity),
        lambda: tf.constant(capacity),
        lambda: replay_size
    )
    batch_indices = tf.random_uniform(
        [batch_size], min_value, max_value, dtype=tf.int32
    )

    # States are N consecutive frames. So we need to convert each index into a
    # vector of the form [index - N + 1, index - N + 2, ..., index]
    batch_slices = tf.map_fn(
        lambda i: modulo(tf.range(i - state_size + 1, i + 1), capacity),
        batch_indices
    )

    return batch_indices, batch_slices


def init_memory(capacity, screen_dims, state_size, batch_size, num_actions,
                frame_input, action_input, reward_input, terminal_input,
                name=None):
    index = tf.Variable(0, name='replay_index')
    size = tf.Variable(0, name='replay_size')
    # index, index_update = init_index(capacity)
    # size, size_update = init_size(capacity)

    batch_indices, batch_slices = get_batch_slices(batch_size, state_size, size, capacity)

    insert_reward, reward_sample = init_rewards(
        capacity, index, reward_input, batch_indices, name
    )
    insert_terminal, terminal_sample = init_terminals(
        capacity, index, terminal_input, batch_indices, name
    )
    insert_action, action_sample = init_actions(
        capacity, index, action_input, batch_indices, num_actions, name
    )

    insert_frame, current_state, state_sample, next_state_sample = init_frames(
        capacity,
        screen_dims,
        state_size,
        index,
        frame_input,
        batch_slices,
        name
    )

    insert_ops = [insert_reward, insert_terminal, insert_action, insert_frame]
    # Ensure that insertions occur before index and size are updated. Without
    # the control dependency, the states of index and size at time of insertion
    # would be nondeterministic.
    with tf.control_dependencies(insert_ops):
        index_update = init_index_update(capacity, index)
        size_update = init_size_update(capacity, size)

    insert = tf.group(
        insert_frame,
        insert_action,
        insert_reward,
        insert_terminal,
        index_update,
        size_update
    )

    return (
        insert,
        current_state,
        state_sample,
        action_sample,
        reward_sample,
        next_state_sample,
        terminal_sample
    )
