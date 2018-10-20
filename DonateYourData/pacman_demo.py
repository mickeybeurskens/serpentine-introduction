

''''
HEre you can play at MsPacman as a player.
The control is by using the arrow keys, or awsd
    - w is up
    - a is left
    - s is down
    - d is right

You can reset the game by pressing 'r'.
You can save the game by pressing 't'.

Have fun :)
'''

import gym
import pickle
import pygame
import random
import numpy as np
from Development.Wouter.DataProcessor.data_io import DataIO

def image_for_pygame(obs):
    obs = np.flip(np.array(obs), axis=0)
    img = pygame.surfarray.make_surface(obs)
    img = pygame.transform.rotate(img, 90)
    return pygame.transform.smoothscale(img, (400, 600))


def preprocess(observation, location):
    prev_observation = observation[0:160, 0:160, 1]  # Set the observation as the previous observation for the next step
    P  = np.array(np.where(prev_observation == 164))
    G1 = np.array(np.where(prev_observation == 184))
    G2 = np.array(np.where(prev_observation == 122))
    G3 = np.array(np.where(prev_observation == 89))
    G4 = np.array(np.where(prev_observation == 72))

    not_found = 0
    if P.size < 2:
        P = location[0:2]
    if G1.size < 2:
        G1 = location[2:4]
        not_found += 1
    if G2.size < 2:
        G2 = location[4:6]
        not_found += 1
    if G3.size < 2:
        G3 = location[6:8]
        not_found += 1
    if G4.size < 2:
        G4 = location[8:10]
        not_found += 1
    if not_found == 4:
        D = np.around([np.average(P[0]), np.average(P[1]), 80, 80, 80, 80, 80, 80, 80, 80, 80, 80])
    else:
        D = np.around([np.average(P[0]), np.average(P[1]),
            np.average(G1[0]), np.average(G1[1]), np.average(G2[0]), np.average(G2[1]),
            np.average(G3[0]), np.average(G3[1]), np.average(G4[0]), np.average(G4[1])])
    return D





## Colors init
#black = (0, 0, 0)
white = (255, 255, 255)

## Data storage init
data = DataIO(gamename='MsPacman-v0', session='PacmanPlayer', save_games=True, iteration=0, use_disk=True)
#game_memory = []
data.new_generation()
data.new_game()

loc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      #Pac/G1/G2/G3/G4




## Environment init
game_name = 'MsPacman-v0'
env_human = gym.make(game_name)
env_human.reset()
env_AI = gym.make(game_name)
env_AI.reset()
action_space = env_human.action_space.__dict__["n"]-1
exit = False
action = random.randint(0, action_space)
if game_name == 'MsPacman-v0':
    for _ in range(50):
        env_human.step(action)
        env_AI.step(action)
score = 0

## Pygame init
display_width = 800
display_height = 600
pygame.init()
game_display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption(f"Atari Game interface: {game_name}")
clock = pygame.time.Clock()



## Loop
while not exit:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                action = 1

            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                action = 2

            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                action = 3

            if  event.key == pygame.K_DOWN or event.key == pygame.K_s:
                action = 4

            if  event.key == pygame.K_p:
                exit = True
                data.save_game(score)
                print('EXITTTT')

            if event.key == pygame.K_r:
                data.save_game(score)
                env_human.reset()
                env_AI.reset()
                score = 0
                data.new_game()

    if exit:
        break

    obs_AI, _, _, _ = env_AI.step(random.randint(0, action_space))
    game_display.blit(image_for_pygame(obs_AI), (400, 0))
    obs, reward, done, info = env_human.step(action)
    score += reward
    data.add_frame(obs, action, reward, info)
    game_display.blit(image_for_pygame(obs), (0, 0))
    #loc = preprocess(obs, loc)
    #print(loc)

    pygame.draw.lines(game_display, white, True, [(display_width//2, 0), (display_width//2, display_height)], 3)
    pygame.display.update()
    clock.tick(25)

data.close()
print('Done')
pygame.quit()
quit()