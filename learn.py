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
