import os
import tensorflow as tf
import numpy as np


class QNetwork():
    """A convolutional neural network approximator for the action-value function
    of a deep Q-learning algorithm. This houses both the behavior and target
    networks.
    """
    def __init__(self, config, num_actions, restore=False):
        print('Constructing QNetwork')

        self.num_actions = num_actions
        self.discount_factor = config['discount_factor']
        self.target_update_freq = config['target_update_freq']
        self.total_updates = 0
        self.checkpoint_path = 'results/checkpoints/{0}/{1}'.format(
            config['game'],
            config['agent_type'].__name__
        )
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.checkpoint_filename = 'agent.ckpt'

        self.behavior_scope = 'behavior'
        self.target_scope = 'target'

        self.build_network(config)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if restore:
            print('Loading Checkpoint...')
            record_path = tf.train.latest_checkpoint(self.checkpoint_path)
            self.saver.restore(self.sess, record_path)
            print('Checkpoint Loaded')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('Network Constructed')

    def build_network(self, config):
        """Build main network with behavior and target networks, loss, and
        optimizer.
        """
        self.observations = tf.placeholder(
            tf.float32,
            shape=[
                None,
                config['screen_dims'][0],
                config['screen_dims'][1],
                config['history_length']
            ],
            name='observation'
        )
        self.actions = tf.placeholder(
            tf.float32,
            shape=[None, self.num_actions],
            name='actions'
        )
        self.rewards = tf.placeholder(
            tf.float32,
            shape=[None],
            name='rewards'
        )
        self.next_observations = tf.placeholder(
            tf.float32,
            shape=[
                None,
                config['screen_dims'][0],
                config['screen_dims'][1],
                config['history_length']
            ],
            name='next_observation'
        )
        self.terminals = tf.placeholder(
            tf.float32,
            shape=[None],
            name='terminals'
        )
        self.normalized_observations = self.observations / 255.0
        self.normalized_next_observations = self.next_observations / 255.0

        self.behavior_q_layer = self.build_action_value_network(
            self.normalized_observations,
            self.behavior_scope,
            True
        )
        self.target_q_layer = self.build_action_value_network(
            self.normalized_next_observations,
            self.target_scope,
            False
        )

        self.loss = self.build_loss(
            config['error_clipping'],
            self.num_actions
        )
        self.train_op = self.build_graves_rmsprop_optimizer(
            config['learning_rate'],
            config['rmsprop_decay'],
            config['rmsprop_epsilon'],
            config['gradient_clip']
        )

    def build_action_value_network(self, observation_pl, scope, trainable):
        """Build the behavior or target network depending on arguments.
        """
        with tf.variable_scope(scope):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                observation_pl, 32, 8, 4,
                activation_fn=tf.nn.relu,
                trainable=trainable
            )
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2,
                activation_fn=tf.nn.relu,
                trainable=trainable
            )
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1,
                activation_fn=tf.nn.relu,
                trainable=trainable
            )

            # Fully connected layers
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(
                flattened,
                512,
                trainable=trainable
            )
            return tf.contrib.layers.fully_connected(
                fc1,
                self.num_actions,
                trainable=trainable
            )

    def build_loss(self, error_clip, num_actions):
        """Build loss graph"""
        with tf.name_scope('loss'):
            predictions = tf.reduce_sum(
                tf.mul(self.behavior_q_layer, self.actions),
                1
            )

            max_action_values = self.build_max_action_values(num_actions)

            targets = tf.stop_gradient(
                self.rewards + (
                    self.discount_factor * max_action_values * (1 - self.terminals)
                )
            )

            diff = tf.abs(predictions - targets)

            if error_clip >= 0:
                clipped = tf.clip_by_value(diff, 0.0, error_clip)
                extra = diff - clipped
                errors = (0.5 * tf.square(clipped)) + (error_clip * extra)
            else:
                errors = (0.5 * tf.square(diff))

            return tf.reduce_sum(errors)

    def build_max_action_values(self, num_actions):
        return tf.reduce_max(self.target_q_layer, 1)

    def build_graves_rmsprop_optimizer(self, learning_rate, rmsprop_decay,
                                       rmsprop_constant, gradient_clip):
        """Alex Graves' RMSprop algorithm:

        https://arxiv.org/abs/1308.0850
        """
        with tf.name_scope('rmsprop'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads = [gv[0] for gv in grads_and_vars]
            params = [gv[1] for gv in grads_and_vars]

            if gradient_clip > 0:
                grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

            avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
                         for var in params]
            avg_square_grads = [tf.Variable(tf.zeros(var.get_shape()))
                                for var in params]

            update_avg_grads = [
                grad_pair[0].assign(
                    (rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1])
                )
                for grad_pair in zip(avg_grads, grads)
            ]
            update_avg_square_grads = [
                grad_pair[0].assign(
                    (rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1]))
                )
                for grad_pair in zip(avg_square_grads, grads)
            ]
            avg_grad_updates = update_avg_grads + update_avg_square_grads

            rms = [
                tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
                for avg_grad_pair in zip(avg_grads, avg_square_grads)
            ]

            rms_updates = [
                grad_rms_pair[0] / grad_rms_pair[1]
                for grad_rms_pair in zip(grads, rms)
            ]
            train = optimizer.apply_gradients(zip(rms_updates, params))

            return tf.group(train, tf.group(*avg_grad_updates))

    def copy_model_parameters(self):
        """Copies the model parameters from target to behavior networks."""
        behavior_params = [t for t in tf.trainable_variables()
                           if t.name.startswith(self.behavior_scope)]
        behavior_params = sorted(behavior_params, key=lambda v: v.name)
        target_params = [t for t in tf.model_variables()
                         if t.name.startswith(self.target_scope)]
        target_params = sorted(target_params, key=lambda v: v.name)

        update_ops = []
        for behavior_v, target_v in zip(behavior_params, target_params):
            operation = target_v.assign(behavior_v)
            update_ops.append(operation)

        self.sess.run(update_ops)

    def predict(self, observation):
        """Get action-value predictions for an observation"""
        result = self.sess.run(
            self.behavior_q_layer,
            feed_dict={self.observations: observation}
        )
        return np.squeeze(result)


    def train(self, observations, actions, rewards, next_observations,
              terminals):
        """Train network on a batch of experiences."""

        loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.observations: observations,
                self.actions: actions,
                self.rewards: rewards,
                self.next_observations: next_observations,
                self.terminals: terminals}
            )[1]

        self.total_updates += 1
        if self.total_updates % self.target_update_freq == 0:
            self.copy_model_parameters()

        return loss

    def save_checkpoint(self, step):
        """Save model checkpoint."""
        path = '{0}/{1}'.format(self.checkpoint_path, self.checkpoint_filename)
        self.saver.save(
            self.sess,
            path,
            global_step=step
        )


class DoubleQNetwork(QNetwork):
    def build_max_action_values(self, num_actions):
        max_actions = tf.to_int32(tf.argmax(self.behavior_q_layer, 1))
        indices = tf.range(0, tf.size(max_actions) * num_actions, num_actions) + max_actions
        return tf.gather(tf.reshape(self.target_q_layer, shape=[-1]), indices)
