# fantasy-football-auction-ai
An agent for doing a fantasy football auction, using the gym fantasy football auction environment

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