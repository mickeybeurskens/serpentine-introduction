"""apprentice_taxigame.py
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
    epochs_max = 200  # Maximum number of time steps
    frame_rate = 0.75  # Frames per second played
    epochs = 0
    state = env.reset()
    done = False

    # Render one game on screen, step by step
    while not done:
        env.render()
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        epochs += 1
        if epochs > epochs_max:
            break
        time.sleep(1 / frame_rate)
    env.render()


def train_taxi(q_table, env):
    """Takes a q table and trains it, returns training data"""
    # Hyper parameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    episodes = 1000

    # For plotting
    all_epochs = []
    all_penalties = []

    # Training loop
    for i in range(episodes):
        # Reset game
        done = False
        epochs, penalties = 0, 0

        while not done:
            break
            # Epsilon decides the rate of random actions

            # Take a step in the taxi environment (e.g. do one of 6 actions)

            # Calculate a new Q value based on the step made in the environment (learning)

            # Some bookkeeping for next simulation step

        # Some text to keep you up to date on the training progress
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

        # Some bookkeeping to store data of each game
        all_epochs.append(epochs)
        all_penalties.append(penalties)

    print("Training finished\n")

    t_data = [all_epochs, all_penalties]
    return t_data


def get_performance(q_table, env):
    """Print the average performance of an agent over 100 episodes"""
    # Some storage and simulation parameters
    total_epochs = 0
    total_penalties = 0
    total_reward = 0
    episodes = 100
    epochs_max = 10000

    # Simulate without randomness for a number of episodes
    for _ in range(episodes):
        # Reset game
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            # Make a step using the q table
            action = np.argmax(q_table[state])  # Best action according to q table
            state, reward, done, info = env.step(action)

            # Bookkeeping
            if reward == -10:
                penalties += 1
            epochs += 1
            total_reward += reward

            # Break if the maximum number of timesteps is reached
            if epochs > epochs_max:
                warnings.warn("Maximum number of time steps reached!", UserWarning)
                break

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    print(f"Average reward per move: {total_reward / total_epochs}\n")


def get_random_performance(env):
    """Shows the performance of a random agent over 100 episodes"""
    # Some storage and simulation parameters
    total_epochs = 0
    total_penalties = 0
    total_reward = 0
    episodes = 100

    # Simulate without randomness for a number of episodes
    for _ in range(episodes):
        # Reset game
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            # Make a step using the q table
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            # Bookkeeping
            if reward == -10:
                penalties += 1

            epochs += 1
            total_reward += reward

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    print(f"Average reward per move: {total_reward / total_epochs}\n")


def save_q_table(q_table):
    """Save a q_table in a pickle file for later use"""
    save_file = open("q_table.pickle", "wb")
    pickle.dump(q_table, save_file)
    save_file.close()


def load_q_table():
    """Reload a saved q_table"""
    load_file = open("q_table.pickle", "rb")
    return pickle.load(load_file)


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