"""
Trains the agent.
"""
import gym
import random
import numpy.random
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.base import InputSpec
from tensorforce.agents import DQNAgent
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.core.networks import Layer
from tensorforce.core.preprocessing import Preprocessor
from tensorforce.execution import Runner
import gym_fantasy_football_auction.envs

from fantasy_football_auction_ai.agents.kerasrl import ConvDQNFantasyFootballAgent

# set random seed so it is reproducible
random.seed(123)
numpy.random.seed(123)


class BatchNormalization(Layer):

    def __init__(self, variance_epsilon=1e-6, scope='batchnorm', summary_labels=None):
        super(BatchNormalization, self).__init__(scope=scope, summary_labels=summary_labels)
        self.variance_epsilon = variance_epsilon

    def tf_apply(self, x, update):
        mean, variance = tf.nn.moments(x, axes=tuple(range(x.shape.ndims - 1)))
        return tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=None,
            scale=None,
            variance_epsilon=self.variance_epsilon
        )


class Permute(Layer):
    def __init__(self, dims, scope='permute', summary_labels=None):
        super(Permute, self).__init__(scope=scope, summary_labels=summary_labels)
        self.dims = tuple(dims)

    def tf_apply(self, x, update):
        return tf.transpose(x, perm=self.dims)

class Conv2DPreprocessor(Preprocessor):
    """
    Adds the channel dimension at the front
    """

    def __init__(self, scope='sequence', summary_labels=()):
        super(Conv2DPreprocessor, self).__init__(scope=scope, summary_labels=summary_labels)

    def reset(self):
        pass

    def tf_process(self, tensor):
        return tf.expand_dims(tensor, 1)

    def processed_shape(self, shape):
        return (1,) + shape


ENV_NAME = 'FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0'
#ENV_NAME = 'FantasyFootballAuction-4OwnerMediumRosterSimpleScriptedOpponent-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

environment = OpenAIGym(gym_id=ENV_NAME)

nb_actions = env.action_space.n
obs_dim = len(env.observation_space.spaces)
obs_dim_2 = env.observation_space.spaces[0].shape

#PR idea notes for tensorforce:
# had to convert my input to float32 from int.
# had to write batchnorm and permute
# can't control ordering for conv2d
# hard to visualize the network architecture and understand it / inputs / outputs of each layer
# have to reshape in order to get my data to work with conv2d (adding channel)
# when there's a problem, hard to know which spot in the network spec was the issue

#make the tf agent
agent = DQNAgent(
    states_spec=dict(type='float32', shape=(obs_dim, obs_dim_2)),
    actions_spec=dict(type='int', num_actions=nb_actions),
    network_spec=[
        dict(type=Permute, dims=(0, 2, 3, 1)),
        dict(type='conv2d', size=5, window=(obs_dim, 1), stride=(obs_dim, 1),
             padding='SAME', l2_regularization=1e-4),
        dict(type=BatchNormalization),
        dict(type=Permute, dims=(0, 2, 3, 1)),
        dict(type='conv2d', size=5, window=(obs_dim, 1), stride=(obs_dim, 1),
             padding='SAME', l2_regularization=1e-4),
        dict(type=BatchNormalization),
        dict(type=Permute, dims=(0, 2, 3, 1)),
        dict(type='conv2d', size=5, window=(obs_dim, 1), stride=(obs_dim, 1),
             padding='SAME', l2_regularization=1e-4),
        dict(type=BatchNormalization),
        dict(type=Permute, dims=(0, 2, 3, 1)),
        dict(type='conv2d', size=5, window=(obs_dim, 1), stride=(obs_dim, 1),
             padding='SAME', l2_regularization=1e-4),
        dict(type=BatchNormalization),
        dict(type='flatten'),
        dict(type='dense',size=nb_actions,activation='softmax')
    ],
    states_preprocessing_spec=[dict(type=Conv2DPreprocessor)],
    batch_size=32,
    optimizer=dict(type='adam', learning_rate=1e-3)
)

runner = Runner(agent=agent,environment=environment)

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)

#agent = ShallowDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0_ShallowDQNFantasyFootballAgent_params.h5f')
#agent = ShallowDQNFantasyFootballAgent(env)
#agent = DQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0_DQNFantasyFootballAgent_params.wip.h5f')
#agent = ConvDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0_ConvDQNFantasyFootballAgent_params.wip.h5f')
#agent = ConvDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-4OwnerMediumRosterSimpleScriptedOpponent-v0_ConvDQNFantasyFootballAgent_params.wip.h5f')

#cProfile.run('agent.learn()', sort='tottime')
#agent.learn(plot=True,train_steps=5000,test_episodes=5)

#DQN learning issues
# things to consider
# fixed inputs which describe the game? (tell it what each player position is and what their value is?)

# it's struggling with bidding rules. Tries to bid when it already bid.
# maybe we should reward it for progressing further in the game? Currently, it sees the same reward regardless of
# how many steps it makes it. So it doesn't really see the difference between making it 50 turns in and making it 1
# turn in. We should probably reward it for not breaking rules.
# Or, we could even punish it every time it breaks a rule but continue with the simulation! It accumulates punishment
# every time it breaks a rule, but it cannot escape the game!
# In this case, we also may want to increase the positive reward for winning to compensate for the very low negative
# reward. Perhaps it earns positive reward based on how long the game generally takes

# Furthermore, the current network might just be too low capacity - 16 neurons may not be enough.

# Another challenge is that it might have already submitted its own bid for the current state and
# then it tries to bid again.

# net needs to be smarter. A few ways to do this.
# Use a different learning alg - try AlphaZero, adapt from ChessZero
# Add some useful features to the observation space
# ChessZero might be an interesting direction because I could possibly get it to be able to play other envs. Good learning
# experience nonetheless. Stand on the shoulders.
# I may need to start it out with some scripted games to bootstrap it to learn the rules. Or it might be able
# to learn fully with self play.
# Might be better to change how action / state space is represented as well. Perhaps action should be dollar amount
# plus player index as a binary

#TODO: Consider a better encoding of the action space so it can be smaller. Look at how they did it in chess.
#TODO: Find better ways to debug the network to see what is happening.
#TODO: Get ChessZero adapted to work here. Can use git submodule and can fork ChessZero as a start so I can modify it
# myself.
#TODO: add an environment without auction - just draft.

#TODO: Consider making a keras-rl style version of AGZ so that it is parameterized. I.e so I can run chess,
# go, or ff on it. In other words, make it able to work in a gym environment.

#TODO: Look at AGZ paper - consider a 3d representation. Look at the architecture. Binary stack rather than
# in sequence / flattened for "state" information for each player.

# What if I can set up an agent for gym which can use the AGZ algorithm?

# I don't like the gym approach. Too impractical. Need to be able to work with the env more directly. But the API
# itself is fine. Just want to mess with the representation.
