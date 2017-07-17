import tensorflow as tf
import tensorflow.contrib.layers as layers


def model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = img_in

        with tf.variable_scope("convnet"):
            # original architecture
            conv_out = layers.convolution2d(conv_out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            conv_out = layers.convolution2d(conv_out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            conv_out = layers.convolution2d(conv_out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = layers.flatten(conv_out)

        # TODO : add policy infer network
        policy_infer_out = img_in

        with tf.variable_scope("policy_infer"):
            policy_infer_out = layers.convolution2d(policy_infer_out, num_outputs=64, kernel_size=6, stride=2, padding='VALID', activation_fn=tf.nn.relu)
            policy_infer_out = layers.convolution2d(policy_infer_out, num_outputs=64, kernel_size=6, stride=2, padding='SAME', activation_fn=tf.nn.relu)
            policy_infer_out = layers.convolution2d(policy_infer_out, num_outputs=64, kernel_size=6, stride=2, padding='SAME', activation_fn=tf.nn.relu)
            policy_infer_out = layers.flatten(policy_infer_out)
            policy_infer_out = layers.fully_connected(policy_infer_out, num_outputs=1024, activation_fn=tf.nn.relu)
            policy_infer_out = layers.fully_connected(policy_infer_out, num_outputs=512, 
                                        activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        policy_infer_out_h = policy_infer_out
        policy_infer_out_y = layers.fully_connected(policy_infer_out_h, num_outputs=num_actions, activation_fn=None)


        with tf.variable_scope("action_value"):
            out = layers.fully_connected(conv_out, num_outputs=512, activation_fn=tf.nn.relu)
            out = tf.concat([out, policy_infer_out_h], axis=1)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out, policy_infer_out_y
