"""
Trains the alphazero-designed agent
"""
import gym
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot
from gym.envs.registration import register

from fantasy_football_auction_ai.agents_selfplay import ConvAlphaZeroFantasyFootballAgent
from fantasy_football_auction_ai.chess_zero.config import Config
from fantasy_football_auction_ai.chess_zero.lib.logger import setup_logger
from fantasy_football_auction_ai.chess_zero.worker import self_play, optimize, evaluate

from logging import getLogger


ENV_NAME='FantasyFootballAuction-4OwnerMediumRosterSelfPlay-v0'
agent_class = ConvAlphaZeroFantasyFootballAgent

#define the agent, which will play from the perspective of each opponent and as the actual agent
selfplay_agent = agent_class('dqn_{}_{}.wip.h5f'.format(ENV_NAME, type(agent_class).__name__))

# create a special gym env, within which the agent can learn via self-play
PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)
register(
    id=ENV_NAME,
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=10.0,

    kwargs={
        'opponents': [selfplay_agent,
                      selfplay_agent,
                      selfplay_agent],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB, RosterSlot.TE, RosterSlot.WRRBTE],
        'starter_value': 1,
        'reward_function': '3'
    }
)
env = gym.make(ENV_NAME)

selfplay_agent.env = env

selfplay_agent.learn()


