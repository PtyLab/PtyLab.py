import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import vectorize
from PtyLab.utils.scanGrids import (
    GenerateRasterGrid,
    GenerateConcentricGrid,
    GenerateNonUniformFermat,
    GenerateFermatSpiral,
)
from PtyLab.utils.scanGrids import tsp_ga, generateTXT


## generate(non - optimal) grid
numPoints = 100  # number of points
radius = 100  # radius of final scan grid (in pixels)
p = 1  # p = 1 is standard Fermat;  p > 1 yields more points towards the center of grid
# first argument: number of points, second argument: scaling of Fermat grid
R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)

xy = np.vstack((R, C)).T  # convert to an (n,2) array
n = len(R)  # number of points
distance = np.sqrt((R[0] - R[-1]) ** 2 + (C[1] - C[-1]) ** 2)
for k in np.arange(1, n):
    distance = distance + np.sqrt((R[k] - R[k - 1]) ** 2 + (C[k] - C[k - 1]) ** 2)
print("initial travel distance: %f" % distance)

# show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(R, C, "o-")
plt.xlabel("um")
plt.title("initial scan grid")
plt.show(block=False)

## traveling salesman problem, genetic algorithm(tsp_ga)
numIteration = 5e3
optRoute = tsp_ga(
    R, C, population_size=40, iterations=numIteration, plotUpdateFrequency=20
).converge()
Rnew = R[optRoute]
Cnew = C[optRoute]


## show optimization result
plt.figure(figsize=(5, 5), num=100)
plt.plot(Rnew, Cnew, "o-")
plt.xlabel("um")
plt.ylabel("um")
plt.title("optimized scan grid")
plt.show(block=False)

averageDistance = np.sum(
    np.sqrt(np.diff(R[optRoute]) ** 2 + np.diff(C[optRoute]) ** 2)
) / len(optRoute)
print("average step size: %i um" % averageDistance)
print("number of scan points: " + str(len(optRoute)))
print("number of scan points: %i" % optRoute.shape[0])

## generate txt file
generateTXT(Rnew, Cnew, fileName="positions_test2")
print("A position file has been saved.")
