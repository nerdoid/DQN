import tensorflow as tf
import numpy as np


capacity = 10
screen_width = 4
screen_height = 4
state_size = 4

sess = tf.Session()
to_wrap = tf.constant(10)
neg_modulo = tf.cond(
    to_wrap >= 0,
    lambda: to_wrap % capacity,
    lambda: to_wrap + capacity
)
print(sess.run(neg_modulo))

#### wrapping counter
index = tf.Variable(4)
index_plus_one = index + 1
index_update = tf.assign(index, index_plus_one % tf.constant(capacity))

size = tf.Variable(0)
size_plus_one = size + 1
next_size = tf.cond(
    tf.equal(size_plus_one, capacity),
    lambda: size,
    lambda: size_plus_one
)
size_update = tf.assign(size, next_size)

sess.run(tf.global_variables_initializer())
print(sess.run(index_update))


#### frame replay
frames = tf.Variable(
    tf.zeros([capacity, screen_height, screen_width], dtype=tf.uint8)
)

cur_state = tf.gather(
    frames,
    tf.range(index - state_size, state_size)
)

sess.run(tf.global_variables_initializer())
print(sess.run(cur_state))
sess.run(index_update)
print(sess.run(cur_state))
print(sess.run(tf.range(index - state_size, index)))



new_frame = np.array(
    [[
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44]
    ]],
    dtype=np.uint8
)

update_frames = tf.scatter_update(
    frames,
    tf.reshape(index, [1]),
    tf.constant(new_frame, dtype=tf.uint8)
)

batch_indices = tf.random_uniform([4], 4 - 1, 10, dtype=tf.int32)
#batch_slices = tf.map_fn(lambda i: tf.range(i - 3, 3), batch_indices)
batch_slices = tf.map_fn(lambda i: tf.range(i - 4 + 1, i + 1), batch_indices)
#prev_slices = (batch_slices - 1) % tf.constant(20)

next_sample = tf.gather(frames, batch_slices)
sample = tf.gather(frames, batch_slices - 1)
print(sess.run(next_sample))
print(sess.run(tf.transpose(next_sample)))

sess.run(tf.global_variables_initializer())

frames_result, _ = sess.run([update_frames, index_update])
print(frames_result)

print(sess.run([batch_indices, sample]))
indices, slices, prev = (sess.run([batch_indices, batch_slices, prev_slices]))
print(indices)
print(slices)
print(prev)

    # terminals_mask = tf.Variable(
    #     tf.zeros([capacity], dtype=tf.bool)
    # )

    # insertion_data = tf.cond(
    #     terminal_pl,
    #     lambda: tf.tile(terminal_tensor, [5]),
    #     lambda: terminal_tensor
    # )
    # insertion_slice = tf.cond(
    #     terminal_pl,
    #     lambda: tf.range(index, index + 5) % capacity,
    #     lambda: tf.reshape(index, [1])
    # )

    # update_terminals_mask = tf.scatter_update(
    #     terminals_mask,
    #     insertion_slice,
    #     insertion_data
    # )

    # with tf.control_dependencies([update_terminals_mask]):


import tensorflow as tf
import numpy as np


def neg_modulo(number, capacity):
    tf.cond(
        number >= 0,
        lambda: number % capacity,
        lambda: number + capacity
    )


def init_rewards(capacity, index, reward_input, batch_slices, name):
    """
    Args:
        reward_input: a tensor with a single reward
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
        batch_slices
    )

    return insert_reward, reward_sample


def init_terminals(capacity, index, terminal_input, batch_slices, name):
    """
    Args:
        terminal_pl: a tensor with a single boolean value
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
        batch_slices
    )

    return insert_terminal, terminal_sample


def init_actions(capacity, index, action_input, batch_slices, name):
    """
    Args:
        action_pl: a tensor with a single action
    """
    actions = tf.Variable(
        tf.zeros([capacity], dtype=tf.uint8)
    )

    insert_action = tf.scatter_update(
        actions,
        tf.reshape(index, [1]),
        action_input
    )

    action_sample = tf.gather(
        actions,
        batch_slices
    )

    return insert_action, action_sample


def get_current_state(index, state_size, frames):
    return tf.gather(
        frames,
        tf.range(index - state_size, index)
    )


def init_frames(capacity, screen_dims, state_size, index, frame_input,
                batch_slices, name=None):

    frames = tf.Variable(
        tf.zeros([capacity, screen_dims[0], screen_dims[1]], dtype=tf.uint8)
    )

    insert_frame = tf.scatter_update(
        frames,
        tf.reshape(index, [1]),
        frame_input
    )

    next_states = tf.gather(
        frames,
        batch_slices
    )
    states = tf.gather(
        frames,
        batch_slices - 1
    )

    current_state = get_current_state(index, state_size, frames)

    return insert_frame, current_state, states, next_states


def init_index_update(capacity, index):
    # Index - will wrap at capacity
    index_plus_one = index + 1
    index_update = tf.assign(
        index,
        index_plus_one % tf.constant(capacity),
        name='replay_index_update'
    )

    return index_update


def init_size_update(capacity, size):
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
    # TODO: A side effect of the below indices is that the wrap-around state_size
    # sequence will never be sampled. Tensorflow's modulo doesn't wrap negative
    # numbers for some reason. So I need to find another way to sample that
    # sequence.
    batch_indices = tf.random_uniform(
        [batch_size], state_size, replay_size, dtype=tf.int32
    )

    # States are N consecutive frames. So we need to convert each index into a
    # vector of the form [index - N + 1, index - N + 2, ..., index]
    batch_slices = tf.map_fn(
        lambda i: tf.range(i - state_size + 1, i + 1),
        batch_indices
    )

    return batch_slices


def init_memory(capacity, screen_dims, state_size, batch_size, frame_input,
                action_input, reward_input, terminal_input, name=None):
    index = tf.Variable(0, name='replay_index')
    size = tf.Variable(0, name='replay_size')
    # index, index_update = init_index(capacity)
    # size, size_update = init_size(capacity)

    batch_slices = get_batch_slices(batch_size, state_size, size, capacity)

    insert_reward, reward_sample = init_rewards(
        capacity, index, reward_input, batch_slices, name
    )
    insert_terminal, terminal_sample = init_terminals(
        capacity, index, terminal_input, batch_slices, name
    )
    insert_action, action_sample = init_actions(
        capacity, index, action_input, batch_slices, name
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



frame_pl = tf.placeholder(
    tf.uint8,
    shape=[1, 4, 4],
    name='observation'
)
action_pl = tf.placeholder(
    tf.uint8,
    shape=[1],
    name='action'
)
reward_pl = tf.placeholder(
    tf.float32,
    shape=[1],
    name='reward'
)
terminal_pl = tf.placeholder(
    tf.bool,
    shape=[1],
    name='terminal'
)


capacity = 10
screen_width = 4
screen_height = 4
state_size = 4
batch_size = 3

memory_ops = init_memory(
    capacity,
    (screen_width, screen_height),
    state_size,
    batch_size,
    frame_pl,
    action_pl,
    reward_pl,
    terminal_pl
)

insert_op, cur_state_op, states, action_sample, reward_sample, next_states, terminal_sample = memory_ops

new_frame = np.array(
    [[
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44]
    ]],
    dtype=np.uint8
)
new_action = np.array([3], dtype=np.uint8)
new_reward = np.array([1.5])
new_terminal = np.array([False])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

insert_result = sess.run(
    insert_op,
    feed_dict={
        frame_pl: new_frame,
        action_pl: new_action,
        reward_pl: new_reward,
        terminal_pl: new_terminal
    }
)

print(sess.run(cur_state_op))

random_states = sess.run(states)
print(random_states)
print(sess.run(action_sample))
print(sess.run(reward_sample))
print(sess.run(next_states))
print(sess.run(terminal_sample))



import tensorflow as tf
import numpy as np


actions = tf.Variable(
    tf.zeros([10], dtype=tf.int32)
)

#actions = np.eye(self.num_actions)[self.actions[samples]]
action_sample = tf.gather(
    actions,
    [0, 3, 5]
)

one_hots = tf.eye(6)[action_sample]

print(sess.run(tf.one_hot(action_sample, 6)))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(action_sample))
print(tf.eye(6)[[3, 2]])
print(sess.run(one_hots))
