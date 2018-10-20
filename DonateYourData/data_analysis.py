import numpy as np
import random
import timeit
import time
from Development.Wouter.DataProcessor.data_io import  DataAnalysis, DataIO
from PIL import Image
from Development.Wouter.DataProcessor.compressors import CompNone, CompWouterGeneric, CompCython160
import matplotlib.pyplot as plt
import cv2


# #################################################################
# #### Usecase 2: Analyzing scores and past training sessions #####
# #################################################################
#
# ## Filtering the data that you want
# # Can use one or more filters if you want
# fsession = ''          # Session filter, exact match
# fsessionsub = 'PacmanPlayer'        # Session filter, substring
# fscoremin = 0                       # Minimum score filter
# fscoremax = 2**60                   # Maximum score filter
# fgame = ''                          # Filter on game name, exact match
# flengthmin = 0                      # Filter on minimum frames in game
# flengthmax = 2**60                  # Filter on maximum frames in game
#
# test = DataAnalysis()
#
# ## Get data specified by keywords. See discord channel for more keywords. Apply filters that you want.
# A, B = test.get_plot_data("Iteration", "Score", fsession=fsession, fsessionsub=fsessionsub, fscoremin=fscoremin)
# # Returns easy to plot stuff :)
#
# print(A)
# print(B)
#
# A, B = test.get_plot_data("Iteration", "Frames", fsession=fsession, fsessionsub=fsessionsub, fscoremin=fscoremin)
#
# print(A)
# print(B)
#
#
#
loadses = DataIO(gamename='TestGame', session='Test', save_games=False, iteration=0, save_session=False, use_disk=False)


loadses.load_session("PacmanPlayer")
A,B,C,D = loadses.get_data()
print('Done')
#
# # LEN = len(A)
# # img = Image.fromarray(A[random.randint(0,LEN)], 'RGB')
# # img.show()
# # img = Image.fromarray(A[random.randint(0,LEN)], 'RGB')
# # img.show()
# # img = Image.fromarray(A[random.randint(0,LEN)], 'RGB')
# # img.show()
#
#
# matrix = A[0:200]
# c = matrix[0]
# c=c.reshape(210, 160, 3) # this is the size of my pictures
# im=plt.imshow(c)
# for row in matrix:
#     row=row.reshape(210, 160, 3) # this is the size of my pictures
#     plt.imshow(row)
#     plt.pause(0.02)
#     plt.show()
#     print('go')
# plt.show()
#



# initialize water image
height = 210
width = 160

# initialize video writer
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 180
video_filename = 'output.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# new frame after each addition of water
for i in range(len(A)):
    out.write(A[i])

# close out the video writer
out.release()