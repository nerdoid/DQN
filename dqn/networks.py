import os
import tensorflow as tf
import numpy as np

class QNetwork():
    """A convolutional neural network approximator for the action-value function
    of a deep Q-learning algorithm. This houses both the behavior and target
    networks.
    """
    def __init__(self, config, num_actions, restore=False, run_name=None):
        print('Constructing QNetwork')

        self.num_actions = num_actions
        self.discount_factor = config['discount_factor']
        self.target_update_freq = config['target_update_freq']
        self.total_updates = 0
        self.checkpoint_path = 'results/checkpoints/{0}/{1}'.format(
            config['game'],
            config['agent_type'].__name__
        )
        if run_name:
            self.checkpoint_path = self.checkpoint_path + '/' + run_name
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
            config['rms_scale'],
            config['rms_constant']
        )

    def build_action_value_network(self, observation_pl, scope, trainable):
        """Build the behavior or target network depending on arguments.
        """
        with tf.variable_scope(scope):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                observation_pl, 32, 8, 4,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                trainable=trainable
            )
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                trainable=trainable
            )
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                trainable=trainable
            )

            # Fully connected layers
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(
                flattened,
                512,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable
            )
            return tf.contrib.layers.fully_connected(
                fc1,
                self.num_actions,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable
            )

    def build_loss(self, error_clip, num_actions):
        """Build loss graph"""
        with tf.name_scope('loss'):
            predictions = tf.reduce_sum(
                tf.multiply(self.behavior_q_layer, self.actions),
                1
            )

            max_action_values = self.build_max_action_values(num_actions)

            targets = tf.stop_gradient(
                self.rewards + (
                    self.discount_factor * max_action_values * (1 - self.terminals)
                )
            )

            errors = self.build_errors(predictions, targets, error_clip)
            return tf.reduce_sum(errors)

    def build_errors(self, behavior_qs, target_qs, error_clip):
        """Let's talk about error clipping. The DeepMind Nature paper has this
        to say:

        "We also found it helpful to clip the error term from the update
        [omitted] to be between -1 and 1. Because the absolute value loss
        function |x| has a derivative of -1 for all negative values of x and a
        derivative of 1 for all positive values of x, clipping the squared
        error to be between -1 and 1 corresponds to using an absolute value
        loss function for errors outside of the (-1,1) interval. This form of
        error clipping further improved the stability of the algorithm."

        But this is confusing and not quite what they do in their Lua code.
        In fact the squared error is always positive, so clipping it to (-1, 1)
        makes no sense. And clipping the inner error would still result in a
        constant value of 1 outside of the clipping region. And that gradient
        would be 0, not 1.

        Instead we want a function that is quadratic between -1 and 1, and
        linear, in particular, |x|, outside of that region. That's what the
        below code achieves.
        """
        diff = tf.abs(target_qs - behavior_qs)

        quadratic = diff
        linear = 0.0
        if error_clip > 0:
            quadratic = tf.clip_by_value(diff, 0.0, error_clip)
            linear = diff - quadratic

        errors = 0.5 * tf.square(quadratic) + error_clip * linear
        return errors

    def build_max_action_values(self, num_actions):
        return tf.reduce_max(self.target_q_layer, 1)

    def build_graves_rmsprop_optimizer(self, learning_rate, rms_scale,
                                       rms_constant):
        """Alex Graves' RMSProp algorithm:

        https://arxiv.org/abs/1308.0850

        The Graves RMSProp has a non-standard rms term in the equation:

        eps
        --- * gradients
        rms

        The strategy here is to manually compute rms, divide the gradients
        element-wise by rms, then feed the resulting gradient tensor into
        TensorFlow's vanilla gradient descent optimizer which handles the eps.
        """
        with tf.name_scope('rmsprop'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss))

            rms_sum = [tf.Variable(tf.zeros(gradient.get_shape()))
                       for gradient in gradients]
            rms_momentum_sum = [tf.Variable(tf.zeros(gradient.get_shape()))
                                for gradient in gradients]

            rms_updates = self.build_rms_updates(
                rms_sum,
                rms_momentum_sum,
                rms_scale,
                gradients
            )

            rms = [
                tf.sqrt(
                    rms_momentum_pair[0] - tf.square(rms_momentum_pair[1]) + (
                        rms_constant
                    )
                )
                for rms_momentum_pair in zip(rms_sum, rms_momentum_sum)
            ]

            scaled_gradients = [
                grad_rms_pair[0] / grad_rms_pair[1]
                for grad_rms_pair in zip(gradients, rms)
            ]
            train = optimizer.apply_gradients(zip(scaled_gradients, variables))

            return tf.group(train, tf.group(*rms_updates))

    def build_rms_updates(self, rms_sum, rms_momentum_sum, rms_scale,
                          gradients):
        rms_sum_updates = [
            rms_and_cur_grads[0].assign(
                rms_scale * rms_and_cur_grads[0] + (
                    (1 - rms_scale) * tf.square(rms_and_cur_grads[1])
                )
            )
            for rms_and_cur_grads in zip(rms_sum, gradients)
        ]
        rms_momentum_sum_updates = [
            rms_and_cur_grads[0].assign(
                rms_scale * rms_and_cur_grads[0] + (
                    (1 - rms_scale) * rms_and_cur_grads[1]
                )
            )
            for rms_and_cur_grads in zip(rms_momentum_sum, gradients)
        ]
        return rms_sum_updates + rms_momentum_sum_updates

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
