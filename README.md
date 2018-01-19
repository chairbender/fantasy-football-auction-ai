# fantasy-football-auction-ai
An agent for doing a fantasy football auction, using the gym fantasy football auction environment

Some code was based on https://github.com/Zeta36/chess-alpha-zero to implement the "alpha zero"
learning agent.

To run:
````
pip install .
python learn.py
````

# Performance

This shows how long it takes to solve each task.

| Task  | Agent | Episodes |
| ------------------------------------------------------------------- | ------------------------------ | ----- |
| FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0 (reward 2.1)  | ShallowDQNFantasyFootballAgent | 423   |
| FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0 (reward 1.1)  | ShallowDQNFantasyFootballAgent | 16658   |
| dqn_FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0 (reward 3)  | ConvDQNFantasyFootballAgent | 1525   |
| dqn_FantasyFootballAuction-4OwnerSmallRosterSimpleScriptedOpponent-v0 (reward 3)  | ConvDQNFantasyFootballAgent | 3939 (only did 1 test episode though)   |
