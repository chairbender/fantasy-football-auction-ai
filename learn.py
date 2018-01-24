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

from fantasy_football_auction_ai.agents.kerasrl import Conv2DQNFantasyFootballAgent

#ENV_NAME = 'FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0'
ENV_NAME = 'FantasyFootballAuction-4OwnerMediumRosterSimpleScriptedOpponent-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

#agent = ShallowDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0_ShallowDQNFantasyFootballAgent_params.h5f')
#agent = ShallowDQNFantasyFootballAgent(env)
#agent = DQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0_DQNFantasyFootballAgent_params.wip.h5f')
#agent = ConvDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0_ConvDQNFantasyFootballAgent_params.wip.h5f')
agent = Conv2DQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-4OwnerMediumRosterSimpleScriptedOpponent-v0_Conv2DQNFantasyFootballAgent_params.wip.h5f')

#cProfile.run('agent.learn()', sort='tottime')
agent.learn(plot=True,train_steps=1000,test_episodes=5)
