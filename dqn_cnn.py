import tensorflow as tf
import numpy as np
import random, gym, math, sys, pickle
from pathlib import Path
from cnn_tf import ConvNeuralNet

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay
UPDATE_TARGET_FREQUENCY = 1000
LEARNING_RATE = 0.00025


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
	def __init__(self, network_config):
		self.observation_shape = network_config["observation_shape"]
		self.nb_actions = network_config["nb_actions"]

        self.steps = 0
        self.nb_features = nb_features
        self.nb_actions = nb_actions
        self.min_epsilon = MIN_EPSILON
        self.max_epsilon = MAX_EPSILON
        self.epsilon = MAX_EPSILON
        self.lmbd = LAMBDA
        self.C = UPDATE_TARGET_FREQUENCY
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.brain = Brain(network_config)
        self.memory = Memory()


	def act(self, s):
		# decides what action to take in state s
		if random.random() < self.epsilon:
			return random.randint(0, self.nb_actions-1)
		else:
			return np.argmax(self.brain.predictOne(s))

	def observe(self, sample):
		# adds sample (s, a, r, s_) to memory replay
        if self.steps % self.C == 0:
            print("TARGET NETWORK UPDATED")
            self.brain.update_target_network()

            # evaluate s_0 somehow TODO
            # S = np.array([8.60847550e-03, -3.64020162e-05, 3.91938297e-02, -2.37661435e-03])
            # pred = agent.brain.predictOne(S)
            # print(pred[0])
            # sys.stdout.flush()

        # adds sample (s, a, r, s_) to memory replay
        self.memory.add(sample)
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lmbd * self.steps)
        self.steps += 1


	def replay(self):
		# replays memories and improves
		batch = self.memory.sample(self.batch_size)

		no_state = np.zeros(self.observation_shape)
		states = np.array([o[0] for o in batch])
		states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])
		q_s = self.brain.predict(states)
		q_s_next_target = self.brain.predict(states_, target_network=True)
		q_s_next_dqn = self.brain.predict(states_, target_network=False) # this is needed for Double DQN

		X = np.zeros((len(batch),) + self.observation_shape)
		y = np.zeros((len(batch), self.nb_actions))

        for i in range(len(batch)):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            target = q_s[i]
            if s_ is None:
                target[a] = r
            else:
                # Double DQN
                best_action_dqn = np.argmax(q_s_next_dqn[i])
                target[a] = r + self.gamma * q_s_next_target[i][best_action_dqn]

            X[i] = s
            y[i] = target

        self.brain.train(X, y)



class RandomAgent:
    def __init__(self, nb_action):
        self.nb_action = nb_action
        self.memory = Memory()

    def act(self, s):
        return random.randint(0, self.nb_action-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass


class Brain:
	def __init__(self, network_config):
		self.model = ConvNeuralNet(network_config)


	def predict(self, states, target_network=False):
		# predicts Q function values for a batch of states
		return self.model.predict(states, target_network)


	def predictOne(self, state):
		# predicts Q function values for one state
		state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
		return self.model.predict(state).flatten()


	def train(self, states, targets):
		# performs training step with batch
		self.model.train_step(states, targets)

	def update_target_network(self):
		# update target with copy of current estimation of NN
		self.model.update_target()


class Memory: # stored as ( s, a, r, s_ )
	def __init__(self):
		self.capacity = MEMORY_CAPACITY
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
	I = I[::2,::2,0] # downsample by factor of 2 and take only the first channel
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float)


if __name__ == "__main__":
	# MAGIC SETTINGS
	PROBLEM = 'Pong-v0'
	max_steps = 10000000000

	# START MAGIC
	env = Environment(PROBLEM, max_steps)
	real_observation_shape = env.env.observation_space.shape
	downscaled_observation_shape = (80, 80, 1)
	nb_actions = env.env.action_space.n

	print("real observation_shape is {}, the agent sees only {}".format(real_observation_shape, downscaled_observation_shape))
	# print(env.env.action_space.n)
	if len(sys.argv) < 3:
		print("usage: python3 dqn.py <nb_epochs> <iter_printed> [<output_file>]")
		sys.exit()
	nb_epochs = int(sys.argv[1])
	iter_printed = int(sys.argv[2])

	# explore first with a random agent
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)
    agent.memory.memory_array = randomAgent.memory.memory_array

    # now train a real agent
	network_config = {
		"observation_shape": downscaled_observation_shape, # TODO check if this is true and if all channels are correctly specified in cnn_tf
		"nb_actions": nb_actions,
		"learning_rate": LEARNING_RATE,
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














