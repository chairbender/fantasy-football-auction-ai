"""
Defines the agents that have been designed to solve fantasy football tasks
"""
import cProfile
import abc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from drawnow import drawnow

import numpy as np
from gym_fantasy_football_auction.envs.agents import FantasyFootballAgent
from keras import Input

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Activation, Conv2D, BatchNormalization, Add, Permute
from keras.optimizers import Adam
from keras.regularizers import l2
from multiprocessing import Lock
from rl.agents import DQNAgent
from rl.callbacks import Callback

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, Model

from fantasy_football_auction_ai.chess_zero.agent.player_chess import VisitStats
from fantasy_football_auction_ai.chess_zero.config import Config
from fantasy_football_auction_ai.chess_zero.configs import normal, mini


class ConvAlphaZeroFantasyFootballAgent(FantasyFootballAgent):
    """
    Like AlphaZero rather than using DQN. Doesn't use Keras-rl

    Plays the actual game of chess, choosing moves based on policy and value network predictions coming
    from a learned model

    Attributes:
        :ivar list: stores info on the moves that have been performed during the game
        :ivar Config config: stores the whole config for how to run
        :ivar PlayConfig play_config: just stores the PlayConfig to use to play the game. Taken from the config
            if not specifically specified.
        :ivar int labels_n: length of self.labels.
        :ivar list(str) labels: all of the possible move labels (like a1b1, a1c1, etc...)
        :ivar dict(str,int) move_lookup: dict from move label to its index in self.labels
        :ivar list(Connection) pipe_pool: the pipes to send the observations of the game to to get back
            value and policy predictions from
        :ivar dict(str,Lock) node_lock: dict from FEN game state to a Lock, indicating
            whether that state is currently being explored by another thread.
        :ivar VisitStats tree: holds all of the visited game states and actions
            during the running of the AGZ algorithm
    """
    def __init__(self, env, config: Config, initial_weights_file=None, pipes=None, play_config=None, dummy=False):
        """

        :param env: Gym environment to learn in. Must support
        :param str initial_weights: optional. path to the h5f file which contains the initial weights.
        """
        super().__init__(env)

        # below arrays store recorded data
        # observation
        self.observations = []
        # policy values per action, calc from visit count
        self.policies = []
        # values, 1 if it led to self winning, 0 if it led to self losing
        self.values = []
        self.initial_weights_file = initial_weights_file

        self.tree = defaultdict(VisitStats)
        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = config.n_labels
        self.labels = config.labels

        if dummy:
            return

        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
        self.model = self.build()
        if self.initial_weights_file is not None:
            try:
                self.model.load_weights(self.initial_weights_file)
            except:
                # just skip loading
                pass

    def act(self, auction, my_idx):
        """
        Invoked when this agent is told to take action as an opponent
        :param auction:
        :param my_idx:
        :return:
        """

        # get an observation from this player's perspective
        observation = self.env.observation_from_perspective(my_idx)

        # determine which action to take
        act = self.action(observation)

        # save the observation and visit counts for training
        # we need to append a tuple - the observation, and
        # a list representing the policy output for each action, given by visit count
        self.observations.append(observation)
        self.policies.append(self.calc_policy(observation))
        # we don't know it yet, just put the index of the owner for future reference
        self.values.append(my_idx)

        # do the action
        return act

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        nb_actions = self.env.action_space.n
        obs_dim = len(self.env.observation_space.spaces)
        obs_dim_2 = self.env.observation_space.spaces[0].shape
        mc = mini.ModelConfig()
        in_x = x = Input((1, obs_dim, obs_dim_2))

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=(obs_dim, 1), strides=(obs_dim, 1), padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-" + str(mc.cnn_first_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)
        # reshape it so it actually makes sense to learn more features on it
        x = Permute((2, 1, 3))(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x
        res_out = Permute((2, 1, 3))(res_out)

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="policy_conv-1-2")(x)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(nb_actions, kernel_regularizer=l2(mc.l2_reg), activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        return Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        mc = mini.ModelConfig()
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=(mc.cnn_filter_num,1), strides=(mc.cnn_filter_num, 1), padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv1-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Permute((2, 1, 3))(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=(mc.cnn_filter_num,1), strides=(mc.cnn_filter_num, 1), padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv2-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = Permute((2, 1, 3))(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def learn(self):
        """
        Iteratively trains and tests until solved (defined as having a winrate of > .99 for test_episodes episodes.

        prints the number of episodes until it was solved

        :param int train_steps: number of steps to train on before each round of testing
        :param int test_episodes: number of episodes to test on before going back to training
        :param boolean plot: whether to visually display a plot of the test winrate over time.
        :return int: number of episodes taken to solve
        """

        # Collect data - play a match via self-play
        env = self.env
        observation = env.reset()
        done = False
        while not done:
            # determine next action and take it
            act = self.action(observation)
            observation, r, d, info = env.step(act)
            done = d
            # save the observation and visit counts for training
            # we need to append a tuple - the observation, and
            # a list representing the policy output for each action, given by visit count
            self.observations.append(observation)
            self.policies.append(self.calc_policy(observation))
            # we don't know it yet, just put the index of the owner for future reference
            self.values.append(0)

        scores = env.auction.scores(env.starter_value)
        max_score = max(scores)
        for i in range(len(scores)):
            if scores[i] == max_score:
                scores[i] = 1
            else:
                scores[i] = -1

        scored_values = [scores[value] for value in self.values]
        self.values = scored_values

        # train
        self.model.model.fit(self.observations, [self.policies, self.values],
                             batch_size=32,
                             epochs=1,
                             shuffle=True,
                             validation_split=0.02)

    def reset(self):
        """
        reset the tree to begin a new exploration of states
        """
        self.tree = defaultdict(VisitStats)

    def action(self, env, perspective_index) -> int:
        """
        Figures out the next best move
        within the specified environment and returns an int encoding of the action

        :param FantasyFootballAuctionEnv env: env to act within
        :param int perspective_index: index of owner whose perspective should be taken
        :return int: the action
        """
        self.reset()

        # do a search to create stats to calculate the policy
        root_value, naked_value = self.search_moves(env, perspective_index)
        # calculate the actual policy
        policy = self.calc_policy(env, perspective_index)
        # TODO: Consider alternate policies - temperature or something
        my_action = int(np.random.choice(range(self.labels_n), p=policy))

        return my_action

    def search_moves(self, env, perspective_index) -> (float, float):
        """
        Looks at all the possible moves using the AGZ MCTS algorithm
         and finds the highest value possible move. Does so using multiple threads to get multiple
         estimates from the AGZ MCTS algorithm so we can pick the best.

        :param FantasyFootballAuctionEnv env: env to search for moves within
        :param int perspective_index: index of owner whose perspective should be taken
        :return (float,float): the maximum value of all values predicted by each thread,
            and the first value that was predicted.
        """
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move, env=env.copy(), perspective_index=perspective_index, is_root_node=True))

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0]  # vals[0] is kind of racy

    def search_my_move(self, env, perspective_index, is_root_node=False) -> float:
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)

        This method searches for possible moves, adds them to a search tree, and eventually returns the
        best move that was found during the search.

        :param FantasyFootballAuctionEnv env: env to search for the move from
        :param int perspective_index: index of owner whose perspective should be taken
        :param boolean is_root_node: whether this is the root node of the search.
        :return float: value of the move. This is calculated by getting a prediction
            from the value network.
        """
        if env.done:
            win_index = env.auction.winning_owner_index()
            if win_index == perspective_index:
                return 1
            return -1

        state = state_key(env, perspective_index)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v  # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, perspective_index, is_root_node)

            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t)
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v

        This gets a prediction for the policy and value of the state within the given env
        :return (float, float): the policy and value predictions for this state
        """
        state_planes = env.canonical_input_planes()

        leaf_p, leaf_v = self.predict(state_planes)
        # these are canonical policy and value (i.e. side to move is "white")

        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p)  # get it back to python-chess form

        return leaf_p, leaf_v

    def predict(self, state_planes, perspective_index):
        """
        Gets a prediction from the policy and value network
        :param state_planes: the observation state represented as planes
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    # @profile
    def select_action_q_and_u(self, env, perspective_index, is_root_node) -> int:
        """
        Picks the next action to explore using the AGZ MCTS algorithm.

        Picks based on the action which maximizes the maximum action value
        (ActionStats.q) + an upper confidence bound on that action.

        :param observation env: env to look for the next moves within
        :param is_root_node: whether this is for the root node of the MCTS search.
        :return int: the move to explore
        """
        # this method is called with state locked
        state = state_key(env, perspective_index)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None:  # push p to edges
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))

        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1 - e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn):
        """
        Applies a random fluctuation to probability of choosing various actions
        :param policy: list of probabilities of taking each action
        :param turn: number of turns that have occurred in the game so far
        :return: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp
            = less.
        """
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env, perspective_index):
        """calc π(a|s0)
        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = state_key(env, perspective_index)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy
def state_key(env, perspective_index) -> str:
    """
    :param FantasyFootballAuctionEnv env: env to encode
    :param int perspective_index: perspective to encode from
    :return str: a str representation of the game state
    """
    #TODO: Use some encoding of the fantasy football auction state
    fen = env.board.fen().rsplit(' ', 1) # drop the move clock
    return fen[0]