import tensorflow as tf
import math

def test1():
	node1 = tf.constant(3.0, tf.float32)
	node2 = tf.constant(4.0)
	print(node1, node2)

	sess = tf.Session()
	print(sess.run([node1, node2]))

	W = tf.Variable([.3], tf.float32)
	b = tf.Variable([-.3], tf.float32)
	x = tf.placeholder(tf.float32)
	linear_model = W * x + b

	init = tf.global_variables_initializer()
	sess.run(init)

	print(sess.run(linear_model, {x:[1,2,3,4]}))

	y = tf.placeholder(tf.float32)
	squared_deltas = tf.square(linear_model - y)
	loss = tf.reduce_sum(squared_deltas)
	print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)

	sess.run(init)
	for i in range(1000):
		sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
	# print(sess.run([W, b]))
	curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:[1,2,3,4], y:[0,-1,-2,-3]})
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


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


def loss(q_value_outputs, q_value_targets):
	""" Calculates the MSE loss from Q value outputs and Q value targets.

	Args:
		q_value_outputs: Q value outputs tensor, float - [batch_size, nb_actions].
		q_value_targets: Q value targets tensor, float - [batch_size, nb_actions].

	Returns:
		loss: Loss tensor, float - [batch_size].
	"""
	mse = tf.nn.l2_loss(
		q_value_outputs - q_value_targets,
		name="mse")
	return mse


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
		self.nb_features = nb_features # = env.env.observations_space.shape[0]
		self.nb_actions = nb_actions # = env.env.action_space.n
		self.hidden_units = hidden_units # = 64
		self.learning_rate = learning_rate # = 0.01


		self.observations_placeholder = tf.placeholder(tf.float32, shape=(None, self.nb_features), name="observations_placeholder")
		self.q_value_targets = tf.placeholder(tf.float32, shape=(None, self.nb_actions))
		self.q_value_outputs, self.dqn_weights = inference("dqn", self.observations_placeholder, self.nb_features, self.nb_actions, self.hidden_units)
		self.loss = loss(self.q_value_outputs, self.q_value_targets)
		self.train_op = train(self.learning_rate, self.loss)

		self.sess = tf.Session()

		self.observations_target_placeholder = tf.placeholder(tf.float32, shape=(None, self.nb_features), name="observations_target_placeholder")
		self.q_value_outputs_target, self.target_weights = inference("target", self.observations_target_placeholder, self.nb_features,
																	 self.nb_actions, self.hidden_units)

		init = tf.global_variables_initializer()
		self.sess.run(init)

	def train_step(self, observations_batch, targets_batch):
		_, loss_value = self.sess.run([self.train_op, self.loss],
					  				  feed_dict={
					      				  self.observations_placeholder: observations_batch,
					      				  self.q_value_targets: targets_batch
					      			  })


	def predict_target(self, observation):
		outputs = self.sess.run(self.q_value_outputs,
					  	 		feed_dict={
					  	  			self.observations_placeholder: observation
					  			})
		return outputs

	def predict(self, observations_batch, target=False):
		if not target:
			outputs = self.sess.run(self.q_value_outputs,
						  	 		feed_dict={
						  	  			self.observations_placeholder: observations_batch
						  			})
		else:
			outputs = self.sess.run(self.q_value_outputs_target,
						  	 		feed_dict={
						  	  			self.observations_target_placeholder: observations_batch
						  			})
		return outputs

	def update_target(self):
		for dqn_w, target_w in zip(self.dqn_weights, self.target_weights):
			copy_w = target_w.assign(dqn_w)
			self.sess.run(copy_w)
		# with self.graph_target.as_default() as g:
			# for w in self.all_weights:

		# tf.assign(self.q_value_outputs_frozen, self.q_value_outputs)