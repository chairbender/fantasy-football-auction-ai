"""
Trains the agent.
"""
import gym
import random
import numpy.random
import numpy as np
import pkg_resources
import tensorflow as tf
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot
from gym.envs import register
from gym_fantasy_football_auction import SimpleScriptedFantasyFootballAgent
from tensorflow.python.layers.base import InputSpec
from tensorforce.agents import DQNAgent
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.core.networks import Layer
from tensorforce.core.preprocessing import Preprocessor
from tensorforce.execution import Runner
import gym_fantasy_football_auction.envs

from fantasy_football_auction_ai.agents.kerasrl import ConvDQNFantasyFootballAgent
PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)

# Some test environments to allow experimenting

#Won in 2135 episodes.
register(
    id='TestEnv-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=5.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(30, 0.9, 0.05)],
        'players': players, 'money': 30,
        'roster': [RosterSlot.QB, RosterSlot.QB, RosterSlot.QB],
        'starter_value': 1,
        'reward_function': '1'
    }
)

#Won in 1700 episodes
register(
    id='TestEnv-v1',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=5.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(30, 0.9, 0.05),
                      SimpleScriptedFantasyFootballAgent(30, 0.9, 0.05)],
        'players': players, 'money': 30,
        'roster': [RosterSlot.QB, RosterSlot.QB, RosterSlot.QB],
        'starter_value': 1,
        'reward_function': '1'
    }
)

#Won in 1986 episodes
register(
    id='TestEnv-v2',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=5.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(30, 0.9, 0.05),
                      SimpleScriptedFantasyFootballAgent(30, 0.9, 0.05),
                      SimpleScriptedFantasyFootballAgent(30, 0.9, 0.05)],
        'players': players, 'money': 30,
        'roster': [RosterSlot.QB, RosterSlot.QB, RosterSlot.QB],
        'starter_value': 1,
        'reward_function': '1'
    }
)

register(
    id='TestEnv-v3',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=5.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(30, 0.5, 0.05),
                      SimpleScriptedFantasyFootballAgent(30, 0.5, 0.05),
                      SimpleScriptedFantasyFootballAgent(30, 0.5, 0.05)],
        'players': players, 'money': 30,
        'roster': [RosterSlot.QB, RosterSlot.QB, RosterSlot.QB],
        'starter_value': 1,
        'reward_function': '1'
    }
)

#performance -

ENV_NAME = 'TestEnv-v3'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

#agent = ShallowDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0_ShallowDQNFantasyFootballAgent_params.h5f')
#agent = ShallowDQNFantasyFootballAgent(env)
#agent = DQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0_DQNFantasyFootballAgent_params.wip.h5f')
#agent = ConvDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0_ConvDQNFantasyFootballAgent_params.wip.h5f')
agent = ConvDQNFantasyFootballAgent(env, 'dqn_{}_ConvDQNFantasyFootballAgent_params.wip.h5f'.format(ENV_NAME), step_through_test=False)

#cProfile.run('agent.learn()', sort='tottime')
agent.learn(plot=True,train_steps=100, test_episodes=10)

#Alternate learning approach - use this instead to ensure that it learns in each environment that
# previous models have been able to learn, after adjusting the model itself
ENV_NAMES = ['TestEnv-v0', 'TestEnv-v1']

#for env_name in ENV_NAMES:
#    env = gym.make(ENV_NAME)
#    agent.learn(plot=True, train_steps=1000, test_episodes=10)
