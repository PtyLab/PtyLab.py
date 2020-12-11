import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import vectorize
from fracPy.utils.scanGrids import GenerateRasterGrid, GenerateConcentricGrid, GenerateNonUniformFermat
from fracPy.utils.scanGrids import tsp_ga


## generate(non - optimal) grid

numPoints = 100   # number of points
radius = 100    # radius of final scan grid (in pixels)
p = 1    # p = 1 is standard Fermat;  p > 1 yields more points towards the center of grid
# first argument: number of points, second argument: scaling of Fermat grid
R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)

xy = np.vstack((R, C)).T  # convert to an (n,2) array
distance = np.sum(np.sqrt(np.diff(R) ** 2 + np.diff(C) ** 2))+\
           np.sqrt((R[0] - R[-1]) ** 2 + (C[1] - C[-1]) ** 2)
print('initial travel distance: %i um' %distance)

# show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(R, C, 'o-')
plt.xlabel('um')
plt.title('initial scan grid')
plt.show(block=False)

## traveling salesman problem, genetic algorithm(tsp_ga)
numIteration = 5e3
optRoute = tsp_ga(xy=xy, population_size=40, iterations=numIteration).converge()
Rnew = xy[optRoute, 0]
Cnew = xy[optRoute, 1]

## show optimization result
plt.figure(figsize=(5, 5), num=100)
plt.plot(Rnew, Cnew, 'o-')
plt.xlabel('um')
plt.ylabel('um')
plt.title('optimized scan grid')
plt.show(block=False)

averageDistance = np.sum(np.sqrt(np.diff(R[optRoute]) ** 2 + np.diff(C[optRoute]) ** 2)) / len(optRoute)
print('average step size: %i um' %averageDistance)
print('number of scan points: '+ str(len(optRoute)))

## generate txt file

# ColsForTXT = [R(optRoute) C(optRoute)]
#
#
# fileID = fopen('positions.txt', 'w') # open
# file
# for writing('w')
#     fprintf(fileID, '#12s #4u\r\n', 'number of positions: ', size(ColsForTXT, 1))
# fprintf(fileID, '#12s #12s\r\n', 'y (row) [um] |', 'x (col) [um]')
# fprintf(fileID, '#4.2f #4.2f\r\n', ColsForTXT)
# fclose(fileID)


