"""
Defines the agents that have been designed to solve fantasy football tasks
"""
import abc
import matplotlib.pyplot as plt
from drawnow import drawnow

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.callbacks import Callback

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

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


class ReinforcementLearningAgent:
    """
    Abstract class for an agent that can be trained and tested on an environment
    """

    def __init__(self, env):
        """
        :param Environment env: gym environment the agent will act in
        """
        self.env = env
        self.train_episodes = 0
        self.total_steps = 0

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

        plot_callbacks = []
        if plot:
           plot_callbacks = [PlotRewardCallback(test_episodes)]
        while True:
            fit_history = agent.fit(self.env, nb_steps=train_steps, verbose=0, visualize=False).history
            self.train_episodes += len(fit_history['episode_reward'])
            self.total_steps += train_steps
            print("Training episodes: " + str(self.train_episodes))
            history = agent.test(self.env, nb_episodes=test_episodes, verbose=0, visualize=False, callbacks=plot_callbacks).history
            # check for it being solved
            if all(reward > .99 for reward in history['episode_reward']):
                print("Solved in " + str(self.train_episodes) + " episodes of training.")
                agent.save_weights('dqn_{}_params.h5f'.format(self.env.spec.id), overwrite=True)
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
        :param str initial_weights: optional. path to the h5f file which contains the initial weights.
        """
        super().__init__(env)

        self.initial_weights_file = initial_weights_file

    def agent(self):
        nb_actions = self.env.action_space.n
        obs_dim = self.env.observation_space.shape
        model = Sequential()
        model.add(Flatten(input_shape=(1, obs_dim)))
        model.add(Embedding(201, 8, input_length=obs_dim))
        model.add(Flatten())
        model.add(Dense(nb_actions, activation='linear'))
        print(model.summary())

        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=256,
                       enable_dueling_network=True,
                       target_model_update=1e-2, policy=BoltzmannQPolicy(), batch_size=128, train_interval=128)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if self.initial_weights_file is not None:
            dqn.load_weights(self.initial_weights_file)

        return dqn
