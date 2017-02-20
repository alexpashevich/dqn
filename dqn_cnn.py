import tensorflow as tf
import numpy as np
import random, gym, math, sys, pickle
from pathlib import Path
from cnn_tf import ConvNeuralNet

class Environment:
	def __init__(self, problem, max_steps):
		self.problem = problem
		self.env = gym.make(problem)
		self.max_steps = max_steps


	def run(self, agent, render=False):
		# runs one episode
		s = prepro(self.env.reset())
		R = 0
		max_stepsize = 500

		for i in range(self.max_steps):
			a = agent.act(s)
			s_, r, done, info = self.env.step(a)
			s_ = prepro(s_)
			if done:
				s_ = None
			agent.observe((s, a, r, s_))
			agent.replay()
			s = s_
			R += r
			if render and i%3==0: self.env.render()
			if done:
				break

		return R


class Agent:
	def __init__(self,
				 network_config,
				 lmbd=0.001,
				 min_epsilon=0.1,
				 max_epsilon=1,
				 batch_size=32,
				 gamma=0.99,
				 C=1000):
		self.observation_shape = network_config["observation_shape"]
		self.nb_actions = network_config["nb_actions"]
		self.min_epsilon = min_epsilon
		self.max_epsilon = max_epsilon
		self.epsilon = max_epsilon
		self.lmbd = lmbd
		self.brain = Brain(network_config)
		self.memory = Memory()
		self.steps = 0
		self.batch_size = batch_size
		self.gamma = gamma
		self.C = C


	def act(self, s):
		# decides what action to take in state s
		if random.random() < self.epsilon:
			return random.randint(0, self.nb_actions-1)
		else:
			return np.argmax(self.brain.predictOne(s))

	def observe(self, sample):
		# adds sample (s, a, r, s_) to memory replay
		self.memory.add(sample)
		self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lmbd * self.steps)
		if self.steps % self.C == 0:
			self.brain.update_target()
		self.steps += 1


	def replay(self):
		# replays memories and improves
		batch = self.memory.sample(self.batch_size)

		no_state = np.zeros(self.observation_shape)
		states = np.array([o[0] for o in batch])
		states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])
		q_s = self.brain.predict(states)
		q_s_next_target = self.brain.predict(states_, target_network=True)
		q_s_next_dqn = self.brain.predict(states_, target_network=False) # this is needed for DDQN

		X = np.zeros((len(batch),) + self.observation_shape)
		y = np.zeros((len(batch), self.nb_actions))

		for i in range(len(batch)):
			s, a, r, s_ = batch[i]
			target = q_s[i]
			if s_ is None:
				target[a] = r
			else:
				# DDQN
				best_action_dqn = np.argmax(q_s_next_dqn[i])
				target[a] = r + self.gamma * q_s_[i][best_action_dqn]
				import pudb; pudb.set_trace() # TODO check
			X[i] = s
			y[i] = target

		self.brain.train(X, y)

class Brain:
	def __init__(self, network_config):
		self.model = ConvNeuralNet(network_config)


	def predict(self, states, target_network=False):
		# predicts Q function values for a batch of states
		return self.model.predict(states, target_network)


	def predictOne(self, state):
		# predicts Q function values for one state
		state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
		return self.model.predict(state)


	def train(self, states, targets):
		# performs training step with batch
		self.model.train_step(states, targets)

	def update_target(self):
		# update target with copy of current estimation of NN
		self.model.update_target()


class Memory: # stored as ( s, a, r, s_ )
	def __init__(self, capacity=1000000):
		self.capacity = capacity
		self.memory_array = []


	def add(self, sample):
		# adds sample to memory
		self.memory_array.append(sample)
		if len(self.memory_array) > self.capacity:
			self.memory_array.pop(0)


	def sample(self, n):
		# return random batch of n samples
		n = min(n, len(self.memory_array))
		return random.sample(self.memory_array, n)


def prepro(I):
	""" prepro 210x160x3 uint8 frame into 80x80x3 float """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float)


if __name__ == "__main__":
	# MAGIC SETTINGS
	PROBLEM = 'Pong-v0'
	max_steps = 500
	learning_rate = 0.001


	# START MAGIC
	env = Environment(PROBLEM, max_steps)
	observation_shape = env.env.observation_space.shape
	nb_actions = env.env.action_space.n

	print("observation_shape is {}".format(observation_shape))
	# print(env.env.action_space.n)
	if len(sys.argv) < 3:
		print("usage: python3 dqn.py <nb_epochs> <iter_printed> [<output_file>]")
		sys.exit()
	nb_epochs = int(sys.argv[1])
	iter_printed = int(sys.argv[2])

	network_config = {
		"observation_shape": (80, 80, 1), # TODO check if this is true and if all channels are correctly specified in cnn_tf
		"nb_actions": nb_actions,
		"learning_rate": learning_rate,
		"filter_size1": 8,
		"filter_size2": 4,
		"filter_size3": 3,
		"nb_ch1": 32,
		"nb_ch2": 64,
		"nb_ch3": 64,
		"stride1": 4,
		"stride2": 2,
		"stride3": 1,
		"fc_units1": 256
	}
	agent = Agent(network_config)
	rewards = []
	for i in range(nb_epochs):
		r = env.run(agent, True if i % iter_printed == 0 and iter_printed > 0 else False)
		rewards.append(r)
		print("[{}] Total reward = {}".format(i, r))


	if len(sys.argv) > 3:
		pickle.dump(rewards, Path("dumps/" + sys.argv[3]).open('wb'))














