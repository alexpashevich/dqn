import tensorflow as tf
import numpy as np
import random, gym, math, sys, pickle
from pathlib import Path
from nn_tf import NeuralNet
import time

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay
UPDATE_TARGET_FREQUENCY = 1000
action_step = 0.1

class Environment:
    def __init__(self, problem, max_steps):
        self.problem = problem
        self.env = gym.make(problem)
        self.max_steps = max_steps


    def run(self, agent, render=False):
        # runs one episode
        s = self.env.reset()
        R = 0

        for i in range(self.max_steps):
            a = agent.act(s)
            s_, r, done, info = self.env.step([a])
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

    def __init__(self, nb_features, nb_actions):
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

        self.brain = Brain(nb_features, nb_actions)
        self.memory = Memory()


    def act(self, s):
        # decides what action to take in state s
        # if random.random() < self.epsilon:
        #     return random.randint(0, self.nb_actions-1)
        # else:
        #     return np.argmax(self.brain.predictOne(s))
        if random.random() < self.epsilon:
            action = random.randint(0, self.nb_actions-1)
        else:
            action = np.argmax(self.brain.predictOne(s))
        if action == 0:
            return action_step
        else:
            return -action_step


    def observe(self, sample):
        if self.steps % self.C == 0:
            print("TARGET NETWORK UPDATED")
            self.brain.update_target_network()

        if self.steps % 10000 == 0:
            S = np.array([8.60847550e-03, -3.64020162e-05, 3.91938297e-02, -2.37661435e-03])
            pred = agent.brain.predictOne(S)
            print(pred[0])
            sys.stdout.flush()

        # adds sample (s, a, r, s_) to memory replay
        self.memory.add(sample)
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lmbd * self.steps)
        self.steps += 1


    def replay(self):
        # replays memories and improves
        batch = self.memory.sample(self.batch_size)
        no_state = np.zeros(self.nb_features)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        q_s = self.brain.predict(states)
        q_s_next_target = self.brain.predict(states_, target=True)
        q_s_next_dqn = self.brain.predict(states_, target=False) # this is needed for Double DQN

        X = np.zeros((len(batch), self.nb_features))
        y = np.zeros((len(batch), self.nb_actions))

        for i in range(len(batch)):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = q_s[i]
            if s_ is None:
                t[a] = r
            else:
                # Double DQN
                best_action_dqn = np.argmax(q_s_next_dqn[i])
                t[a] = r + self.gamma * q_s_next_target[i][best_action_dqn]

            X[i] = s
            y[i] = t

        self.brain.train(X, y)


class RandomAgent:
    def __init__(self, nb_action):
        self.nb_action = nb_action
        self.memory = Memory()

    def act(self, s):
        action = random.randint(0, self.nb_action-1)
        if action == 0:
            return action_step
        else:
            return -action_step

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass


class Brain:
    def __init__(self, nb_features, nb_actions, hidden_units=64, learning_rate=0.00025):
        self.model = NeuralNet(nb_features, nb_actions, hidden_units, learning_rate)


    def predict(self, states, target=False):
        # predicts Q function values for a batch of states
        return self.model.predict(states, target)


    def predictOne(self, state, target=False):
        # predicts Q function values for one state
        state = state.reshape(1, -1)
        return self.model.predict(state, target).flatten()


    def train(self, states, targets):
        # performs training step with batch
        self.model.train_step(states, targets)


    def update_target_network(self):
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


    def is_full(self):
        return len(self.memory_array) == self.capacity


if __name__ == "__main__":
    PROBLEM = 'Pendulum-v0' #CartPole-v0
    max_steps = 10000
    env = Environment(PROBLEM, max_steps)
    nb_features = env.env.observation_space.shape[0]
    print(type(env.env.action_space))
    # nb_actions = env.env.action_space.n
    nb_actions = env.env.action_space.shape[0] * 2
    rewards = []

    if len(sys.argv) < 3:
        print("usage: python3 dqn_nn.py <nb_epochs> <iter_printed> [<output_file>]")
        sys.exit()
    nb_epochs = int(sys.argv[1])
    iter_printed = int(sys.argv[2])

    agent = Agent(nb_features, nb_actions)
    randomAgent = RandomAgent(nb_actions)

    # explore first with a random agent
    while randomAgent.memory.is_full() == False:
        env.run(randomAgent)
    agent.memory.memory_array = randomAgent.memory.memory_array

    # train the dqn agent
    for i in range(nb_epochs):
        r = env.run(agent, True if i % iter_printed == 0 and iter_printed > 0 else False)
        rewards.append(r)
        print("[{}] Total reward = {}".format(i, r))

    if len(sys.argv) > 3:
        pickle.dump(rewards, Path("dumps/" + sys.argv[3]).open('wb'))














