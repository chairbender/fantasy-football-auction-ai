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
| FFEnv1-v0 (reward 3)  | ConvDQNFantasyFootballAgent | 469   |

