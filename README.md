# fantasy-football-auction-ai
NOTE: This project is indefinitely on hold for the forseeable future. Stuff I learned:
* Deep RL is not a super mature field - there are libraries that I expect would exist right now
but which don't actually exist!
* Got some experience building and debugging an RL agent / network architecture
* Got experience building an RL environment (the fantasy football api) / working with python
* It's quite hard to figure out why something is or isn't learning
* Deep RL is not magic (yet), and actually might not be the best approach for solving certain problems. See
 (Deep RL Doesn't Work Yet)[https://www.alexirpan.com/2018/02/14/rl-hard.html]
* It's hard to even get a reliable behavior on the SAME architecture / hyperparameters / environment,
running multiple times. Randomness plays quite a big part.
* Overall, there's lots of hype for Deep RL but it's not quite there yet.
* I suspect that my toy problem may be better solved by other methods, if I actually cared about 
solving it rather than just using it as a way to learn this stuff.
* It's quite hard to find best practices / principles / well-worn paths / "giants shoulders" within
Deep RL. The people who know how to do it well seem to mostly have that info in their heads. 
For Deep RL, we don't seem to have good resources on design and practice unless you spend a lot of
time keeping up on the research and have the knowledge to be able to synthesize that info. This
isn't the sort of thing where a smart programmer can just read a blog post, grab a library off the
shelf, implement some adapters, and let the library do the rest. It's not even comparable to ML -
there's actually quite a lot of info on ML in general, concerning design and practice.
* So, I'm updating my expectations for applying Deep RL to problems. It's not as simple as you
might think, coming from a "Deep RL is magic" perspective. You see the successes and not the failures.
And you don't see how much time goes into things you wouldn't think about.
* Just as an example, here's some things that I didn't expect to spend so much time on: figuring out what
is going on inside my model, figuring out if my environment observations have sufficient info for learning,
figuring out if there's a bug in my environment observations. I think just trying to figure out
and debug why an agent isn't succeeding is probably the majority of the time one would spend on
a Deep RL project. It's hard to figure out what deep nets are doing. If there is a good way to do this,
I didn't find one.
* Maybe in 5 years, this will be much easier.

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

