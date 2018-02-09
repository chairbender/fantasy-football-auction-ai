"""
Defines agents which use the kerasrl library
"""
import abc
import matplotlib.pyplot as plt
from drawnow import drawnow

import numpy as np
from keras import Input

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, BatchNormalization, Add, Permute
from keras.optimizers import Adam
from keras.regularizers import l2
from rl.agents import DQNAgent
from rl.callbacks import Callback

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, Model, BoltzmannGumbelQPolicy


class PlotAllCallback(Callback):
    """
    Keras-nl callback to use which plots what happens over time
    """

    def make_fig(self):
        plt.subplot(2,1,1)
        plt.scatter(self.x, self.y_reward)
        plt.title('Avg Test Reward per ' + str(self.every) + ' Episodes')
        plt.xlabel('Test Batch (' + str(self.every) + ' Episodes per Batch)')
        plt.ylabel('Avg Reward During Batch')
        plt.subplot(2, 1, 2)
        plt.scatter(self.x, self.y_wins)
        plt.title('Winrate per ' + str(self.every) + ' Episodes')
        plt.xlabel('Test Batch (' + str(self.every) + ' Episodes per Batch)')
        plt.ylabel('Winrate During Batch')

    def __init__(self, every=100, max=200):
        """

        :param every: add a point after this number of episodes
        :param max:  max number of points to keep in the plot at one time
        """
        super().__init__()
        self.cumulative_reward = 0
        self.ep_count = 0
        self.x = [0.]
        self.y_reward = [0.]
        self.y_wins = [0.]
        self.every = every
        self.max = max
        self.wins = 0
        plt.ion()


    def on_episode_end(self, episode, logs={}):
        self.ep_count += 1
        self.cumulative_reward += logs['episode_reward']
        if self.env.is_winner():
            self.wins += 1
        if self.ep_count % self.every == 0:
            if len(self.x) >= self.max:
                del self.x[0]
                del self.y_wins[0]
                del self.y_reward[0]
            self.x.append(self.ep_count / self.every)
            self.y_reward.append(self.cumulative_reward / self.every)
            self.y_wins.append(self.wins / self.every)
            drawnow(self.make_fig)
            self.cumulative_reward = 0
            self.wins = 0

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

class StepThroughCallback(Callback):
    """
    Callback to check the game state while testing
    """

    def on_step_end(self, step, logs={}):
        self.env.render()
        input("Press Enter key to continue...\n\n")

class InformedBoltzmannGumbelQPolicy(BoltzmannGumbelQPolicy):
    """
    Just like BoltzmannGumbelQPolicy, but ignores actions which are illegal
    """
    def __init__(self, env, C=1.0):
        """

        :param FantasyFootballAuctionEnv: gym environment to use to check for legal moves using
            .action_legality()
        :param tau: see parent
        :param clip: see parent
        """
        super(self.__class__, self).__init__(C)
        self.env = env

    def select_action(self, q_values):
        # We can't use BGE during testing, since we don't have access to the
        # action_counts at the end of training.
        assert self.agent.training, "BoltzmannGumbelQPolicy should only be used for training, not testing"

        assert q_values.ndim == 1, q_values.ndim
        q_values = q_values.astype('float64')

        # If we are starting training, we should reset the action_counts.
        # Otherwise, action_counts should already be initialized, since we
        # always do so when we begin training.
        if self.agent.step == 0:
            self.action_counts = np.ones(q_values.shape)
        assert self.action_counts is not None, self.agent.step
        assert self.action_counts.shape == q_values.shape, (self.action_counts.shape, q_values.shape)

        beta = self.C / np.sqrt(self.action_counts)
        Z = np.random.gumbel(size=q_values.shape)

        perturbation = beta * Z
        perturbed_q_values = q_values + perturbation
        legality = (np.array(self.env.action_legality()) - 1) * 999
        perturbed_q_values = np.add(perturbed_q_values, legality)
        action = np.argmax(perturbed_q_values)

        self.action_counts[action] += 1
        return action

class InformedGreedyQPolicy(GreedyQPolicy):
    """
    Just like GreedyQPolicy but ignores invalid actions
    """
    def __init__(self, env):
        self.env = env

    def select_action(self, q_values):
        assert q_values.ndim == 1
        # since it's only doing argmax, we need this array to make large negative numbers
        # for invalid actions
        legality = (np.array(self.env.action_legality()) - 1) * 999
        q_values = np.add(q_values, legality)
        action = np.argmax(q_values)
        return action


class KerasRLAgent:
    """
    Abstract class for an agent that can be trained and tested on an environment
    """

    def __init__(self, env, skip_training=False, step_through_test=False):
        """
        :param Environment env: gym environment the agent will act in
        :param boolean skip_training: if true, will test only, no training
        :param boolean step_through_test: if true, during test, pauses for input after each action taken by the agent
            and prints the game state
        """
        self.env = env
        self.train_episodes = 0
        self.total_steps = 0
        self.skip_training = skip_training
        self.step_through_test = step_through_test

    @abc.abstractmethod
    def agent(self):
        """
        create and initialize the keras-nl agent

        :return Agent: the fully initialized keras-nl agent, compiled, ready to test and train. Weights should
            not be initialized from existing models - they should be at their initial settings.
        """
        pass


    def learn(self, train_steps=1000, test_episodes=5, plot=True):
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
            test_callbacks = [PlotAllCallback(test_episodes)]
        winrate_callback = CheckWinrateCallback()
        test_callbacks.append(winrate_callback)
        if self.step_through_test:
            test_callbacks.append(StepThroughCallback())
        while True:
            if not self.skip_training:
                fit_history = agent.fit(self.env, nb_steps=train_steps, verbose=1, visualize=False).history
                self.train_episodes += len(fit_history['episode_reward'])
                self.total_steps += train_steps
                print("Training episodes: " + str(self.train_episodes))
            winrate_callback.reset()
            agent.test(self.env, nb_episodes=test_episodes, verbose=2, visualize=False, callbacks=test_callbacks)
            agent.save_weights('dqn_{}_{}_params.wip.h5f'.format(self.env.spec.id, type(self).__name__), overwrite=True)
            # check for it being solved
            if winrate_callback.winrate() > .99:
                print("Solved in " + str(self.train_episodes) + " episodes of training.")
                agent.save_weights('dqn_{}_{}_params.h5f'.format(self.env.spec.id, type(self).__name__), overwrite=True)
                plt.show()
                break


class ShallowDQNFantasyFootballAgent(KerasRLAgent):
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
                       target_model_update=1e-2, policy=InformedBoltzmannGumbelQPolicy(self.env),
                       test_policy=InformedGreedyQPolicy(self.env), batch_size=128, train_interval=128)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if self.initial_weights_file is not None:
            dqn.load_weights(self.initial_weights_file)
            self.train_episodes = 0

        return dqn

class DQNFantasyFootballAgent(KerasRLAgent):
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
                       target_model_update=1e-2, policy=InformedBoltzmannGumbelQPolicy(self.env),
                       test_policy=InformedGreedyQPolicy(self.env), batch_size=128, train_interval=128)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if self.initial_weights_file is not None:
            try:
                dqn.load_weights(self.initial_weights_file)
            except:
                # just skip loading
                pass

        return dqn


class ConvDQNFantasyFootballAgent(KerasRLAgent):
    """
    A DQN
    """
    def __init__(self, env, initial_weights_file=None, step_through_test=False):
        """
        :param env: Gym environment to learn in
        :param str initial_weights: optional. path to the h5f file which contains the initial weights.
        """
        super().__init__(env, step_through_test=step_through_test)

        self.initial_weights_file = initial_weights_file

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        nb_actions = self.env.action_space.n
        obs_dim = len(self.env.observation_space.spaces)
        obs_dim_2 = self.env.observation_space.spaces[0].shape
        in_x = x = Input((1, obs_dim, obs_dim_2))

        # (batch, channels, height, width)
        x = Conv2D(filters=256, kernel_size=(obs_dim, 1), strides=(obs_dim, 1), padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(1e-4),
                   name="input_conv-5-256")(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)
        # reshape it so it actually makes sense to learn more features on it
        x = Permute((2, 1, 3))(x)

        for i in range(7):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for Q value output
        x = Permute((2,1,3))(res_out)
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(1e-4),
                   name="policy_conv-1-2")(x)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(nb_actions, kernel_regularizer=l2(1e-4), activation="softmax",
                           name="policy_out")(x)

        return Model(in_x, [policy_out], name="chess_model")

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=256, kernel_size=(256, 1), strides=(256, 1), padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Permute((2, 1, 3))(x)
        x = Conv2D(filters=256, kernel_size=(256,1), strides=(256, 1), padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = Permute((2, 1, 3))(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def agent(self):
        nb_actions = self.env.action_space.n
        model = self.build()
        print(model.summary())

        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=32,
                       enable_dueling_network=True,
                       target_model_update=1e-2, policy=InformedBoltzmannGumbelQPolicy(self.env),
                       test_policy=InformedGreedyQPolicy(self.env), batch_size=32, train_interval=32)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if self.initial_weights_file is not None:
            try:
                dqn.load_weights(self.initial_weights_file)
            except:
                # just skip loading
                pass

        return dqn