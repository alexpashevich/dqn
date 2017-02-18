import tensorflow as tf
import numpy as np
import random, gym, math
from nn_tf import NeuralNet
import sys

class Environment:
	def __init__(self, problem):
		self.problem = problem
		self.env = gym.make(problem)


	def run(self, agent, render=False):
		# runs one episode
		s = self.env.reset()
		R = 0
		t = 0

		while True:
			a = agent.act(s)
			s_, r, done, info = self.env.step(a)
			if done:
				s_ = None
			agent.observe((s, a, r, s_))
			agent.replay()
			s = s_
			R += r
			if render and t%3==0: self.env.render()
			t += 1
			if done:
				break

		print("Total reward = {}".format(R))


class Agent:
	def __init__(self,
				 nb_features,
				 nb_actions,
				 lmbd=0.001,
				 min_epsilon=0.01,
				 max_epsilon=1,
				 batch_size=64,
				 hidden_units=64,
				 learning_rate=0.00025,
				 gamma=0.99,
				 C=10):
		self.nb_features = nb_features
		self.nb_actions = nb_actions
		self.min_epsilon = min_epsilon
		self.max_epsilon = max_epsilon
		self.epsilon = max_epsilon
		self.lmbd = lmbd
		self.brain = Brain(nb_features, nb_actions, hidden_units, learning_rate)
		self.memory = Memory()
		self.steps = 0
		self.batch_size = batch_size
		self.gamma = gamma
		self.counter = 0
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
		self.steps += 1


	def replay(self):
		# replays memories and improves
		batch = self.memory.sample(self.batch_size)

		no_state = np.zeros(self.nb_features) # TODO: does it harm or not?
		states = np.array([o[0] for o in batch])
		states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])
		q_s = self.brain.predict(states)
		q_s_ = self.brain.predict(states_, target=False)

		X = np.zeros((len(batch), self.nb_features))
		y = np.zeros((len(batch), self.nb_actions))

		for i in range(len(batch)):
			s, a, r, s_ = batch[i]
			target = q_s[i]
			if s_ is None:
				target[a] = r
			else:
				target[a] = r + self.gamma * np.amax(q_s_[i])
			X[i] = s
			y[i] = target

		self.brain.train(X, y)
		self.counter += 1

		if self.counter % self.C == 0:
			self.brain.update_target()

class Brain:
	def __init__(self, nb_features, nb_actions, hidden_units, learning_rate):
		self.model = NeuralNet(nb_features, nb_actions, hidden_units, learning_rate)


	def predict(self, states, target=False):
		# predicts Q function values for a batch of states
		return self.model.predict(states, target)


	def predictOne(self, state):
		# predicts Q function values for one state
		state = state.reshape(1, -1)
		return self.model.predict(state)


	def train(self, states, targets):
		# performs training step with batch
		self.model.train_step(states, targets)

	def update_target(self):
		# update target with copy of current estimation of NN
		self.model.update_target()


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


if __name__ == "__main__":
	PROBLEM = 'CartPole-v0'
	env = Environment(PROBLEM)
	nb_features = env.env.observation_space.shape[0]
	nb_actions = env.env.action_space.n
	nb_steps = 1000
	iter_printed = int(sys.argv[1])

	agent = Agent(nb_features, nb_actions)
	for i in range(nb_steps):
		env.run(agent, True if i % iter_printed == 0 and iter_printed > 0 else False)












