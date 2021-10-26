import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn='/Users/juliaxu/Documents/F2021/CS330/project/dataset/orbit_benchmark_224/validation/P189/headband/clean/P189--headband--clean--e6PQORY1AVwfRcSNiDFjdCmigw8YT8lcUE6fJ5X97A8/'
pathOut = 'video.mp4'

fps = 30
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort()

#for sorting the file names properly
for i in range(len(files)):
    filename=pathIn + files[i]

    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()