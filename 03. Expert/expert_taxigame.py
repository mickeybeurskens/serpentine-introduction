"""expert_taxigame.py
        A basic introduction into the concepts of machine learning. This script acts as a template.

    Author: Mickey Beurskens
    Based heavily on the tutorial on basic reinforcement learning by Satwik Kansal and Brendan Martin.
    URL: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
"""

import gym
import random
import numpy as np
import warnings
import time
import pickle
from IPython.display import clear_output


def play_game(q_table, env):
    """Visually play a game of Taxi-V2 using a q table"""


def train_taxi():
    """Takes a q table and trains it, returns training data"""


def get_performance():
    """Print the average performance of an agent over 100 episodes"""
    pass


def get_random_performance():
    """Shows the performance of a random agent over 100 episodes"""
    pass


def save_q_table():
    """Save a q_table in a pickle file for later use"""
    pass


def load_q_table():
    """Reload a saved q_table"""
    pass


# Initialization of environment and necessary variables
env = gym.make("Taxi-v2").env
env.reset()

# Import a pre-trained q_table
q_tab = np.zeros([env.observation_space.n, env.action_space.n])
q_tab = load_q_table()

# Visually play a game of Taxi in the console
play_game(q_tab, env)

# Show average performance over 100 games
get_performance(q_tab, env)