"""
Defines the agents that have been designed to solve fantasy football tasks
"""
import abc
import matplotlib.pyplot as plt
from drawnow import drawnow

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Activation
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.callbacks import Callback

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, GreedyQPolicy


class PlotRewardCallback(Callback):
    """
    Keras-nl callback to use which plots reward over time
    """

    def make_fig(self):
        plt.scatter(self.x, self.y)
        plt.title('Total Test Reward per ' + str(self.every) + ' Episodes')
        plt.xlabel('Test Batch (' + str(self.every) + ' Episodes per Batch)')
        plt.ylabel('Total Reward During Batch')

    def __init__(self, every=100, max=200):
        """

        :param every: add a point after this number of episodes
        :param max:  max number of points to keep in the plot at one time
        """
        super().__init__()
        self.cumulative_reward = 0
        self.ep_count = 0
        self.x = [0.]
        self.y = [0.]
        self.every = every
        self.max = max
        plt.ion()

    def on_episode_end(self, episode, logs={}):
        self.ep_count += 1
        self.cumulative_reward += logs['episode_reward']
        if self.ep_count % self.every == 0:
            if len(self.x) >= self.max:
                del self.x[0]
                del self.y[0]
            self.x.append(self.ep_count / self.every)
            self.y.append(self.cumulative_reward)
            drawnow(self.make_fig)
            self.cumulative_reward = 0

class CheckWinrateCallback(Callback):
    """
    Callback to check the winrate during testing
    """
    def __init__(self):
        self.episode_count = 0
        self.win_count = 0

    def on_episode_end(self, episode, logs={}):
        self.episode_count += 1
        if self.env.is_winner():
            self.win_count += 1

    def winrate(self):
        return 0 if self.episode_count == 0 else self.win_count / self.episode_count

    def reset(self):
        self.episode_count = 0
        self.win_count = 0

class InformedBoltzmannQPolicy(BoltzmannQPolicy):
    """
    Just like BoltzmannQPolicy, but ignores actions which are illegal
    """
    def __init__(self, env, tau=1., clip=(-500., 500.)):
        """

        :param FantasyFootballAuctionEnv: gym environment to use to check for legal moves using
            .action_legality()
        :param tau: see parent
        :param clip: see parent
        """
        super(self.__class__, self).__init__(tau, clip)
        self.env = env

    def select_action(self, q_values):
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        exp_values = np.multiply(exp_values, self.env.action_legality())
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

class InformedGreedyQPolicy(GreedyQPolicy):
    """
    Just like GreedyQPolicy but ignores invalid actions
    """
    def __init__(self, env):
        self.env = env

    def select_action(self, q_values):
        assert q_values.ndim == 1
        q_values = np.multiply(q_values, self.env.action_legality())
        action = np.argmax(q_values)
        return action


class ReinforcementLearningAgent:
    """
    Abstract class for an agent that can be trained and tested on an environment
    """

    def __init__(self, env, skip_training=False):
        """
        :param Environment env: gym environment the agent will act in
        :param boolean skip_training: if true, will test only, no training
        """
        self.env = env
        self.train_episodes = 0
        self.total_steps = 0
        self.skip_training = skip_training

    @abc.abstractmethod
    def agent(self):
        """
        create and initialize the keras-nl agent

        :return Agent: the fully initialized keras-nl agent, compiled, ready to test and train. Weights should
            not be initialized from existing models - they should be at their initial settings.
        """
        pass


    def learn(self, train_steps=1000, test_episodes=100, plot=True):
        """
        Iteratively trains and tests until solved (defined as having a winrate of > .99 for test_episodes episodes.

        prints the number of episodes until it was solved

        :param int train_steps: number of steps to train on before each round of testing
        :param int test_episodes: number of episodes to test on before going back to training
        :param boolean plot: whether to visually display a plot of the test winrate over time.
        :return int: number of episodes taken to solve
        """

        agent = self.agent()

        test_callbacks = []
        if plot:
            test_callbacks = [PlotRewardCallback(test_episodes)]
        winrate_callback = CheckWinrateCallback()
        test_callbacks.append(winrate_callback)
        while True:
            if not self.skip_training:
                fit_history = agent.fit(self.env, nb_steps=train_steps, verbose=0, visualize=False).history
                self.train_episodes += len(fit_history['episode_reward'])
                self.total_steps += train_steps
                print("Training episodes: " + str(self.train_episodes))
            winrate_callback.reset()
            agent.test(self.env, nb_episodes=test_episodes, verbose=2, visualize=True, callbacks=test_callbacks)
            agent.save_weights('dqn_{}_{}_params.wip.h5f'.format(self.env.spec.id, type(self).__name__), overwrite=True)
            # check for it being solved
            if winrate_callback.winrate() > .99:
                print("Solved in " + str(self.train_episodes) + " episodes of training.")
                agent.save_weights('dqn_{}_{}_params.h5f'.format(self.env.spec.id, type(self).__name__), overwrite=True)
                plt.show()
                break


class ShallowDQNFantasyFootballAgent(ReinforcementLearningAgent):
    """
    A shallow DQN (if that makes any sense).

    Capable of solving FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0
    """
    def __init__(self, env, initial_weights_file=None):
        """

        :param env: Gym environment to learn in
        :param str initial_weights: optional. path to the h5f file which contains the initial weights. Will skip
            training if this is the case (assumes you just want to test the trained agent)
        """
        super().__init__(env, initial_weights_file is not None)

        self.initial_weights_file = initial_weights_file

    def agent(self):
        nb_actions = self.env.action_space.n
        obs_dim = self.env.observation_space.shape
        model = Sequential()
        model.add(Flatten(input_shape=(1, obs_dim)))
        model.add(Dense(nb_actions, activation='linear'))
        print(model.summary())

        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=256,
                       enable_dueling_network=True,
                       target_model_update=1e-2, policy=InformedBoltzmannQPolicy(self.env),
                       test_policy=InformedGreedyQPolicy(self.env), batch_size=128, train_interval=128)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if self.initial_weights_file is not None:
            dqn.load_weights(self.initial_weights_file)
            self.train_episodes = 0

        return dqn

class DQNFantasyFootballAgent(ReinforcementLearningAgent):
    """
    A DQN
    """
    def __init__(self, env, initial_weights_file=None):
        """

        :param env: Gym environment to learn in
        :param str initial_weights: optional. path to the h5f file which contains the initial weights.
        """
        super().__init__(env)

        self.initial_weights_file = initial_weights_file

    def agent(self):
        nb_actions = self.env.action_space.n
        obs_dim = len(self.env.observation_space.spaces)
        obs_dim_2 = self.env.observation_space.spaces[0].shape
        model = Sequential()
        model.add(Flatten(input_shape=(1, obs_dim, obs_dim_2)))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions, activation='linear'))
        print(model.summary())

        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=256,
                       enable_dueling_network=True,
                       target_model_update=1e-2, policy=InformedBoltzmannQPolicy(self.env),
                       test_policy=InformedGreedyQPolicy(self.env), batch_size=128, train_interval=128)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if self.initial_weights_file is not None:
            dqn.load_weights(self.initial_weights_file)

        return dqn
