import tensorflow as tf
import math
import numpy as np


class ConvNeuralNet(object):
    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, name="weights")
        return tf.Variable(initial)


    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, name="bias")
        return tf.Variable(initial)


    def __conv2d(X, W):
        return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')


    def __max_pool_2x2(X):
        return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    def __inference(self, namespace, observations, network_config):
        all_weights = []

        with tf.name_scope(namespace + '/conv1'):
            filter_size = network_config["filter_size1"]
            nb_ch_in = network_config["observation_shape"][2]
            nb_ch_out = network_config["nb_ch1"]
            stride = network_config["stride1"]
            W_conv = self.__weight_variable([filter_size, filter_size, nb_ch_in, nb_ch_out])
            b_conv = self.__bias_variable([nb_ch_out])
            hidden_conv = tf.nn.relu(tf.nn.conv2d(observations, W_conv, strides=[1,stride,stride,1], padding='SAME') + b_conv)
            # hidden_pool = self.__max_pool_2x2(hidden_conv)
            all_weights.append(W_conv)
            all_weights.append(b_conv)

        with tf.name_scope(namespace + '/conv2'):
            filter_size = network_config["filter_size2"]
            nb_ch_in = network_config["nb_ch1"]
            nb_ch_out = network_config["nb_ch2"]
            stride = network_config["stride2"]
            W_conv = self.__weight_variable([filter_size, filter_size, nb_ch_in, nb_ch_out])
            b_conv = self.__bias_variable([nb_ch_out])
            hidden_conv = tf.nn.relu(tf.nn.conv2d(hidden_conv, W_conv, strides=[1,stride,stride,1], padding='SAME') + b_conv)
            # hidden_pool = self.__max_pool_2x2(hidden_conv)
            all_weights.append(W_conv)
            all_weights.append(b_conv)

        with tf.name_scope(namespace + '/conv3'):
            filter_size = network_config["filter_size3"]
            nb_ch_in = network_config["nb_ch2"]
            nb_ch_out = network_config["nb_ch3"]
            stride = network_config["stride3"]
            W_conv = self.__weight_variable([filter_size, filter_size, nb_ch_in, nb_ch_out])
            b_conv = self.__bias_variable([nb_ch_out])
            hidden_conv = tf.nn.relu(tf.nn.conv2d(hidden_conv, W_conv, strides=[1,stride,stride,1], padding='SAME') + b_conv)
            strides_factor = network_config["stride1"] * network_config["stride2"] * network_config["stride3"]
            out_width = network_config["observation_shape"][0] // strides_factor
            hidden_conv_flat = tf.reshape(hidden_conv, [-1, out_width*out_width*network_config["nb_ch3"]])
            # hidden_pool = self.__max_pool_2x2(hidden_conv)
            all_weights.append(W_conv)
            all_weights.append(b_conv)

        with tf.name_scope(namespace + '/fc1'):
            input_fc = int(out_width ** 2 * network_config["nb_ch3"]) # should be 6400
            output_fc = network_config["fc_units1"]
            W_fc = tf.Variable(
                tf.truncated_normal([input_fc, output_fc],
                                    stddev=1.0 / math.sqrt(float(input_fc))),
                name="weights")
            b_fc = tf.Variable(
                tf.zeros([output_fc]),
                name="biases")
            hidden_fc = tf.nn.relu(tf.matmul(hidden_conv_flat, W_fc) + b_fc)
            all_weights.append(W_fc)
            all_weights.append(b_fc)

        with tf.name_scope(namespace + '/output'):
            input_fc = network_config["fc_units1"]
            output_fc = network_config["nb_actions"]
            W_fc = tf.Variable(
                tf.truncated_normal([input_fc, output_fc],
                                    stddev=1.0 / math.sqrt(float(input_fc))),
                name="weights")
            b_fc = tf.Variable(
                tf.zeros([output_fc]),
                name="biases")
            q_value_outputs = tf.matmul(hidden_fc, W_fc) + b_fc
            all_weights.append(W_fc)
            all_weights.append(b_fc)

        return q_value_outputs, all_weights


    def __loss(self):
        mse = tf.nn.l2_loss(
            self.q_value_outputs - self.q_value_targets,
            name="mse")
        return mse


    def __train(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(self.loss, global_step=global_step)
        return train_op


    def __init__(self, network_config):
        self.learning_rate = network_config["learning_rate"]
        self.observation_shape = network_config["observation_shape"]
        self.nb_actions = network_config["nb_actions"]


        self.observations_placeholder = tf.placeholder(tf.float32, shape=((None,) + self.observation_shape), name="observations_placeholder")
        self.q_value_targets = tf.placeholder(tf.float32, shape=(None, self.nb_actions), name="q_value_targets")
        self.q_value_outputs, self.dqn_weights = self.__inference("dqn", self.observations_placeholder, network_config)
        self.loss = self.__loss()
        self.train_op = self.__train()

        self.sess = tf.Session()

        self.q_value_outputs_target, self.target_weights = self.__inference("target", self.observations_placeholder, network_config)

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def train_step(self, observations_batch, targets_batch):
        _, loss_value = self.sess.run([self.train_op, self.loss],
                                      feed_dict={
                                          self.observations_placeholder: observations_batch,
                                          self.q_value_targets: targets_batch
                                      })


    def predict(self, observations_batch, target=False):
        if target is False:
            outputs = self.sess.run(self.q_value_outputs,
                                    feed_dict={
                                        self.observations_placeholder: observations_batch
                                    })
        else:
            outputs = self.sess.run(self.q_value_outputs_target,
                                    feed_dict={
                                        self.observations_placeholder: observations_batch
                                    })
        return outputs


    def update_target(self):
        for dqn_w, target_w in zip(self.dqn_weights, self.target_weights):
            copy_w = target_w.assign(dqn_w)
            self.sess.run(copy_w)






