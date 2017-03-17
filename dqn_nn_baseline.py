# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# 
# author: Jaromir Janisch, 2016


#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  

import random, numpy, math, gym, sys, pickle
from pathlib import Path

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel() 

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss=hubert_loss, optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

# MEMORY_CAPACITY = 100000
# BATCH_SIZE = 256

# GAMMA = 0.99

# MAX_EPSILON = 1
# MIN_EPSILON = 0.1
# LAMBDA = 0.001      # speed of decay

# UPDATE_TARGET_FREQUENCY = 5000


# ACTION_MIN = -2.0
# ACTION_MAX = 2.0
# ACTION_NUMBER = 20

q_values = []

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.actionCnt-1)
        else:
            action_idx = numpy.argmax(self.brain.predictOne(s))
        return action_idx

        # if random.random() < self.epsilon:
        #     action = random.randint(0, self.actionCnt-1)
        # else:
        #     action = np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            print("TARGET NETWORK UPDATED")
            self.brain.updateTargetModel()

        if self.steps % 10000 == 0:
            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
            # S = numpy.array([-0.33522294, 0.94213883, 0.05842767])
            pred = agent.brain.predictOne(S)
            print(pred[0])
            q_values.append(pred[0])
            sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    # def act(self, s):
    #     return random.randint(0, self.actionCnt-1)

    def act(self, s):
        # all_actions = range(ACTION_MIN, ACTION_MAX, (ACTION_MAX - ACTION_MIN) / ACTION_NUMBER) + (ACTION_MAX - ACTION_MIN) / ACTION_NUMBER / 2
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0

        # while True:
        # all_actions = np.array([(ACTION_MAX - ACTION_MIN) / ACTION_NUMBER * i for i in range(ACTION_NUMBER)]) + ACTION_MIN + (ACTION_MAX - ACTION_MIN) / ACTION_NUMBER / 2

        for i in range(1000000000):
            # self.env.render()

            a = agent.act(s)

            # if a == 0:
            #     s_, r, done, info = self.env.step([action_step])
            # else:
            #     s_, r, done, info = self.env.step([-action_step])
            

            # s_, r, done, info = self.env.step([all_actions[a]])
            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        # print("Total reward:", R)
        return R

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0' # 'Pendulum-v0' #'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n
# actionCnt = ACTION_NUMBER

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

nb_epochs = int(sys.argv[1])
rewards = []

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    for i in range(nb_epochs):
        r = env.run(agent)
        rewards.append(r)
        print("[{}] Total reward = {}".format(i, r))
        if i % 50 == 0:
            pickle.dump([rewards, q_values], Path("dumps/baseline.pkl").open('wb'))
finally:
    # agent.brain.model.save("cartpole-dqn.h5")
    pickle.dump([rewards, q_values], Path("dumps/baseline.pkl").open('wb'))




