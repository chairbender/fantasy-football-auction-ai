"""
For developer testing.
"""
import numpy as np
import gym
import gym_fantasy_football_auction

from drawnow import drawnow

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent, CEMAgent
from rl.callbacks import Callback

from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.policy import BoltzmannQPolicy

import matplotlib.pyplot as plt


class TestResults(Callback):
    wincount = 0
    turncount = 1

    def on_episode_end(self, episode, logs={}):
        if logs['episode_reward'] >= .999:
            self.wincount += 1
            print('\nwon a match!')
        self.turncount += 1
        print('\nwinrate: ' + str(self.wincount / self.turncount))
        self.env.render(mode='human')

class LogTesting(Callback):

    def make_fig(self):
        plt.scatter(self.x, self.y)

    def __init__(self, every, max=200):
        super().__init__()
        self.wincount = 0
        self.turncount = 1
        self.x = [0.]
        self.y = [0.]
        self.i = 1
        self.every = every
        self.max = max
        plt.ion()

    def on_episode_end(self, episode, logs={}):
        if logs['episode_reward'] >= .999:
            self.wincount += 1
        self.turncount += 1
        if self.turncount % self.every == 0:
            if self.i >= self.max:
                del self.x[0]
                del self.y[0]
            self.x.append(self.i)
            self.y.append(self.wincount / self.turncount)
            drawnow(self.make_fig)
            self.turncount = 0
            self.wincount = 0
            self.i += 1

class DebugCallback(Callback):
    def on_action_begin(self, action, logs={}):
        print("\nObservation: " + str(self.model.recent_observation))
        print("Action: " + str(action))

ENV_NAME = 'FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0'
#ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
#env_my = gym.make('FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0')
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape


### CEM Version
def do_cem():
    model = Sequential()
    model.add(Flatten(input_shape=(1,env.observation_space.shape)))
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))
    print(model.summary())

    memory = EpisodeParameterMemory(limit=1000, window_length=1)

    cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000,
                   batch_size=512, train_interval=512, elite_frac=0.05)
    cem.compile()
    # After training is done, we save the best weights.
    cem.load_weights('cem_FF_params.h5f'.format(ENV_NAME))

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    log = LogTesting(1000)
    while True:
        cem.fit(env, nb_steps=1000, verbose=0, visualize=False, callbacks=[log])
        # no need to really test atm since the sim is pretty determnistic for a given set of weights
        # dqn.test(env, nb_episodes=100, verbose=0, visualize=False)
        #
        cem.save_weights('cem_FF_params.h5f'.format(ENV_NAME), overwrite=True)

### DQN Version
def do_dqn():
    # Next, we build a very simple model regardless of the dueling architecture
    # if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
    # Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
    model = Sequential()
    model.add(Flatten(input_shape=(1,env.observation_space.shape)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000, window_length=1)

    # Notes for myself.
    # How this works - it plays the game to get experiences. The actual training runs by sampling from those experiences
    # (stored in the memory var, the most recent 50000 experiences).
    # Then the new policy is used to try again and generate new experiences.
    # params:
    # nb_actions - tells the model how many different actions there are.
    # memory - what to use to store the experiences for experience replay (I think)
    # nb_steps_warmup - how long to run at the very start of training in order to generate enough experience samples for
    #   sampling without replacement. You shouldn't start training until there's "enough" experiences to learn from.
    # target_model_update - idk, some hyperparam
    # policy - the actual policy it uses to make choices based on the learned model
    # batch size - during training, it gets a sample of this size from the experience memory, calculates the error for
    #   all the examples in that batch, then updates the model.
    #   BTW One pass through the entire set of data (the memory) is called an epoch.
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, enable_dueling_network=True,
                   target_model_update=1e-2, policy=BoltzmannQPolicy(), batch_size=128, train_interval=1)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('dqn_FF_params.h5f'.format(ENV_NAME))

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    log = LogTesting(1000)
    while True:
        dqn.fit(env, nb_steps=1000, verbose=0, visualize=False, callbacks=[log])
        # no need to really test atm since the sim is pretty determnistic for a given set of weights
        #dqn.test(env, nb_episodes=100, verbose=0, visualize=False)
        #
        dqn.save_weights('dqn_FF_params.h5f'.format(ENV_NAME), overwrite=True)


#do_cem()
do_dqn()

plt.show()