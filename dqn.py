import tensorflow as tf
import random
import numpy as np

class Environment:
	def run():
		# runs one episode
		s = env.reset()
		while True:
			a = agent.act(s)
			s_, r, done, info = env.step(a)

			if done:
				s_ = None

			agent.observe((s, a, r, s_))
			smth_sub_word
			agent.replay()

			s = s_

			if done:
				break


class Agent:
	def __init__(self, nb_features, nb_actions, lmbd=0.001, min_epsilon=0.01, max_epsilon=1, batch_size=64, gamma=0.99):
		self.nb_features = nb_features
		self.nb_actions = nb_actions
		self.min_epsilon = min_epsilon
		self.max_epsilon = max_epsilon
		self.epsilon = max_epsilon
		self.lmbd = lmbd
		self.brain = Brain()
		self.memory = Memory()
		self.steps = 0
		self.batch_size = batch_size
		self.gamma = gamma


	def act(self, s):
		# decides what action to take in state s
		if random.random < self.epsilon:
			return random.randint(0, self.nb_actions-1)
		else:
			return np.argmax(self.brain.predictOne(s))

	def observe(self, sample):
		# adds sample (s, a, r, s_) to memory replay
		self.memory.add(sample)
		self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lmbd * self.steps)
		self.steps += 1


	def replay(self):
		# replays memories and improves
		batch = self.memory.sample(self.batch_size)

		no_state = np.zeros(self.nb_features) # TODO: does it harm or not?
		states = np.array([o[0] for o in batch])
		states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])
		q_s = self.brain.predict(states)
		q_s_ = self.brain.predict(states_)

		X = np.zeros((len(batch), self.nb_features))
		y = np.zeros((len(batch), self.nb_actions))

		for s, a, r, s_ in batch:
			target = q_s[i]
			if s_ is None:
				target[a] = r
			else:
				target[a] = r + self.gamma * np.amax(q_s_[i])
			X[i] = s
			y[i] = target

		self.brain.train(X, y)

class Brain:
	def __init__(self):
		self.model = ...


	def predict(self, s):
		# predicts Q function values in state s
		self.model.predict(s)


	def train(batch):
		# performs training step with batch
		self.model.fit()


class Memory: # stored as ( s, a, r, s_ )
	def __init__(self, capacity=100000):
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