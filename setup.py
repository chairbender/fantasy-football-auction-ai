from setuptools import setup

setup(name='fantasy_football_auction_ai',
      version='0.0.1',
      install_requires=['gym>=0.7.4', 'fantasy_football_auction>=0.9.6', 'gym_fantasy_football_auction>=0.0.1',
                        'keras-rl>=0.4.0', 'numpy>=1.13.1', 'h5py>=2.7.1'],
      packages=['fantasy_football_auction_ai'],
      package_dir={'fantasy_football_auction_ai': 'fantasy_football_auction_ai'},
      package_data={'fantasy_football_auction_ai': ['data/*.csv']},
      )
