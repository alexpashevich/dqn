import tensorflow as tf
import math
import numpy as np
from tensorflow.python import debug as tf_debug


def inference(namespace, observations, nb_features, nb_actions, hidden_units):
	""" Build a feedforward neural network with 1 hidden layer.

	Args:
		observations: observations placeholder tensor, float - [batch_size, nb_features].
		nb_features: number of features in an observations, float - 0D.
		nb_actions: number of actions, float - 0D.
		hidden_units: number of hidden units, float - 0D.

	Returns:
		q_value_outputs: Output tensor of the NN, float - [batch_size, nb_actions].
	"""
	all_weights = []
	# with graph.as_default():
	with tf.name_scope(namespace + '/hidden'):
		weights = tf.Variable(
			tf.truncated_normal([nb_features, hidden_units],
								stddev=1.0 / math.sqrt(float(nb_features))),
			name="weights")
		biases = tf.Variable(
			tf.zeros([hidden_units]),
			name="biases")
		hidden = tf.nn.relu(tf.matmul(observations, weights) + biases)
		all_weights.append(weights)
		all_weights.append(biases)

	with tf.name_scope(namespace + '/output'):
		weights = tf.Variable(
			tf.truncated_normal([hidden_units, nb_actions],
								stddev=1.0 / math.sqrt(float(hidden_units))),
			name="weights")
		biases = tf.Variable(
			tf.zeros([nb_actions]),
			name="biases")
		q_value_outputs = tf.matmul(hidden, weights) + biases
		all_weights.append(weights)
		all_weights.append(biases)
	return q_value_outputs, all_weights


def clipped_error(x):
	# Huber loss
	try:
		return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
	except:
		return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def loss(q_value_outputs, q_value_targets):
	""" Calculates the MSE loss from Q value outputs and Q value targets.

	Args:
		q_value_outputs: Q value outputs tensor, float - [batch_size, nb_actions].
		q_value_targets: Q value targets tensor, float - [batch_size, nb_actions].

	Returns:
		loss: Loss tensor, float - [batch_size].
	"""
	# loss = tf.nn.l2_loss(q_value_outputs - q_value_targets, name="mse")
	loss = tf.reduce_mean(clipped_error(q_value_targets - q_value_outputs), name='loss', axis=1) # TODO: change to axis=1
	return loss


def train(learning_rate, loss):
	""" Set up training op given the loss and the learning rate.
	
	Args:
		learning_rate: learning rate value, float - 0D.
		loss: Loss tensor, float - [batch_size].

	Returns:
		train_op: The op for training.

	"""
	optimizer = tf.train.RMSPropOptimizer(learning_rate)
	global_step = tf.Variable(0, name="global_step", trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


class NeuralNet(object):
	def __init__(self, nb_features, nb_actions, hidden_units, learning_rate):
		self.nb_features = nb_features
		self.nb_actions = nb_actions
		self.hidden_units = hidden_units
		self.learning_rate = learning_rate


		self.observations_placeholder = tf.placeholder(tf.float32, shape=(None, self.nb_features), name="observations_placeholder")
		self.q_value_targets = tf.placeholder(tf.float32, shape=(None, self.nb_actions))
		self.q_value_outputs, self.dqn_weights = inference("dqn", self.observations_placeholder, self.nb_features, self.nb_actions, self.hidden_units)
		self.loss = loss(self.q_value_outputs, self.q_value_targets)
		self.train_op = train(self.learning_rate, self.loss)
		self.q_value_outputs_target, self.target_weights = inference("target", self.observations_placeholder, self.nb_features,
																	 self.nb_actions, self.hidden_units)
		init = tf.global_variables_initializer()
		# tf.get_default_graph().finalize()
		self.sess = tf.Session()
		self.sess.run(init)

	def train_step(self, observations_batch, targets_batch):
		_, loss_value = self.sess.run([self.train_op, self.loss],
					  				  feed_dict={
					      				  self.observations_placeholder: observations_batch,
					      				  self.q_value_targets: targets_batch
					      			  })



	def predict(self, observations_batch, target_network=False):
		if target_network is False:
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

	def test_output(self):
		observ_rand = np.array([[1,2,3,4], [4,3,2,1], [3,2,4,5]])
		outputs_1 = self.sess.run(self.q_value_outputs,
					  	 		feed_dict={
					  	  			self.observations_placeholder: observ_rand
					  			})
		outputs_2 = self.sess.run(self.q_value_outputs_target,
					  	 		feed_dict={
					  	  			self.observations_placeholder: observ_rand
					  			})
		print("outputs_1.mean() = {}, outputs_2.mean() = {}".format(outputs_1.mean(), outputs_2.mean()))


	def update_target(self):
		for dqn_w, target_w in zip(self.dqn_weights, self.target_weights):
			copy_w = target_w.assign(dqn_w)
			self.sess.run(copy_w)





