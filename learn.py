"""
Trains the agent.
"""
import gym
import random
import numpy.random
import gym_fantasy_football_auction
from fantasy_football_auction_ai.agents import ShallowDQNFantasyFootballAgent, DQNFantasyFootballAgent

# set random seed so it is reproducible
random.seed(123)
numpy.random.seed(123)

#ENV_NAME = 'FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0'
ENV_NAME = 'FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n
obs_dim = env.observation_space.shape

#agent = ShallowDQNFantasyFootballAgent(env,'dqn_FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0_ShallowDQNFantasyFootballAgent_params.h5f')
#agent = ShallowDQNFantasyFootballAgent(env)
agent = DQNFantasyFootballAgent(env)
agent.learn(plot=True)
