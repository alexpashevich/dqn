import tensorflow as tf

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


def inference(observations, nb_features, nb_actions, hidden1_units):
	""" Build a feedforward neural network with 1 hidden layer.

	Args:
		observations: observations placeholder tensor, float - [batch_size, nb_features].
		nb_features: number of features in an observations, float - 0D.
		nb_actions: number of actions, float - 0D.
		hidden1_units: number of hidden units, float - 0D.

	Returns:
		q_value_outputs: Output tensor of the NN, float - [batch_size, nb_actions].
	"""
	with tf.name_scope('hidden1'):
		weights = tf.Variable(
			tf.truncated_normal([nb_features, hidden1_units],
								stddev=1.0 / math.sqrt(float(nb_features))),
			name="weights")
		biases = tf.Variable(
			tf.zeros([hidden1_units]),
			name="biases")
		hidden1 = tf.nn.relu(tf.matmul(observations, weights) + biases)

	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(
			tf.truncated_normal([hidden1_units, nb_actions],
								stddev=1.0 / math.sqrt(float(hidden1_units)))
			name="weights")
		biases = tf.Variable(
			tf.zeros([nb_actions]),
			name="biases")
		q_value_outputs = tf.matmul(hidden1, weights) + biases
	return q_value_outputs


def loss(q_value_outputs, q_value_targets):
	""" Calculates the MSE loss from Q value outputs and Q value targets.

	Args:
		q_value_outputs: Q value outputs tensor, float - [batch_size, nb_actions].
		q_value_targets: Q value targets tensor, float - [batch_size, nb_actions].

	Returns:
		loss: Loss tensor, float - [batch_size].
	"""
	mse = rf.nn.l2_loss(
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
	tf.scalar_summary(loss.op.name, loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name="global_step", trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def run_training():
	nb_features = env.env.observations_space.shape[0]
	nb_actions = env.env.action_space.n
	nb_steps = 1000
	hidden1_units = 64
	learning_rate = 0.01


	with tf.Graph().as_default():
		states_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nb_features))
		q_value_targets = tf.placeholder(tf.float32, shape=(batch_size, nb_actions))
		q_value_outputs = inference(states_placeholder, nb_features, nb_actions, hidden1_units)
		loss = loss(q_value_outputs, q_value_targets)
		train_op = train(learning_rate, loss)
		# summary = tf.summary.merge_all()

		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		for step in xrange(nb_steps):
			sess.run([train_op, loss])

if __name__ == "__main__":
	PROBLEM = 'CartPole-v0'
	env = Environment(PROBLEM)



	features_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nb_features))
	

	