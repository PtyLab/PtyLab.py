import numpy as np
from numpy import vectorize
import random
from matplotlib import pyplot as plt


def GenerateNonUniformFermat(n, radius=1000, power=1):
    """
    generate spiral patterns
    :param n: number of points generated
    :param radius: radius in micrometer
    :param power = 1 is standard Fermat, power>1 yields more points towards the center of grid
    :return:
    R: row
    C: column
    """
    # golden ratio
    r = np.sqrt(np.arange(0, n) / n)
    theta0 = 137.508 / 180 * np.pi
    theta = np.arange(0, n) * theta0
    C = radius * r**power * np.cos(theta)
    R = radius * r**power * np.sin(theta)
    return R, C


def GenerateFermatSpiral(n, c):
    """
    generate Fermat Spiral
    :param n: number of points generated
    :param c: optional argument that controls scaling of spiral
    :return:
    R: row
    C: column
    """
    # golden ratio
    r = np.sqrt(np.arange(0, n)) * c
    theta0 = 137.508 / 180 * np.pi
    theta = np.arange(0, n) * theta0
    C = r * np.cos(theta)
    R = r * np.sin(theta)
    return R, C


def GenerateConcentricGrid(Nr, s, rend):
    """
    generate concentric circles
    :param Nr: number of circles (or shells)
    :param s: number of pixels between points on each circle, roughly calculated as rend/Nr
    :param rend: end radius size (in pixel units)
    :return:
    R: row
    C: column
    """
    dx = 1  # max Resolution (Schritt von einem zum anderen Pixel)
    rstart = dx
    r = np.linspace(rstart, rend, Nr)
    # determine number of positions on k'th shell
    nop = np.zeros(Nr, dtype=int)
    for k in np.arange(Nr):
        nop[k] = int(np.floor(2 * np.pi * r[k] / s))
    positions = np.zeros((sum(nop) + 1, 2))
    ind = 1
    for k in np.arange(1, Nr):
        dtheta = 2 * np.pi / nop[k]
        theta = np.arange(1, nop[k] + 1) * dtheta + 2 * np.pi / (k + 1)
        for l in np.arange(nop[k]):
            positions[ind, :] = r[k] * np.array([np.cos(theta[l]), np.sin(theta[l])])
            ind += 1
    positions = (np.floor(positions / dx)).astype(int)

    R = positions[:, 0]
    C = positions[:, 1]
    return R, C


def GenerateRasterGrid(n, ds, randomOffset=False, amplitude=1):
    """
    generate a raster grid containing n*n points with a period of ds in pixelsize,
    with the option of adding randomOffsets.
    :param n: number of points per dimension
    :param ds: period (# of pixels) per dimension
    :param randomOffset: optional to add random offsets, default: False
    :param amplitude: amplitude for the random offsets, default: 1
    :return:
    R: row
    C: column
    """
    I, J = np.meshgrid(np.arange(n), np.arange(n))
    C = I.reshape(n**2) * ds
    R = J.reshape(n**2) * ds

    # center the scan grid at [0,0]
    # for even numbers
    if np.mod(n, 2) == 0:
        C = C - n * ds / 2
        R = R - n * ds / 2
    # for odd numbers
    else:
        C = C - (n - 1) * ds / 2
        R = R - (n - 1) * ds / 2

    if randomOffset:
        C = C + np.round(amplitude * (-1 + 2 * np.random.rand(C.shape)))
        R = R + np.round(amplitude * (-1 + 2 * np.random.rand(R.shape)))

    R = np.round(R - np.mean(R)).astype(int)
    C = np.round(C - np.mean(C)).astype(int)

    return R, C


def generateTXT(R, C, fileName="position"):
    ColsForTXT = np.vstack((R, C)).T
    fileID = open(fileName + ".txt", "w")
    stringForFile = "number of positions: {} \n".format(ColsForTXT.shape[0])
    fileID.write(stringForFile)
    fileID.write("y (row) [um] | x (col) [um] \n")
    for ii in range(ColsForTXT.shape[0]):
        stringForFile = "{} {} \n".format(ColsForTXT[ii, 0], ColsForTXT[ii, 1])
        fileID.write(stringForFile)
    fileID.close()


class tsp_ga:
    """
    genetic algorithm for traveling salesman problem.
    """

    def __init__(
        self, R, C, start="0", population_size=5, iterations=100, plotUpdateFrequency=20
    ):
        self.xy = np.vstack((R, C)).T
        self.start = start
        self.population_size = population_size
        self.iterations = int(iterations)
        self.plotUpdateFrequency = plotUpdateFrequency

        self.figure = plt.figure(num=999, clear=True, figsize=(5, 5))
        self.ax_scanGridOpt = self.figure.add_subplot(111)
        self.ax_scanGridOpt.set_title("optimized scan grid")
        self.ax_scanGridOpt.set_xlabel("um")
        self.ax_scanGridOpt.set_ylabel("um")
        (self.ax_scanGridOpt_plot,) = self.ax_scanGridOpt.plot(
            self.xy[:, 0], self.xy[:, 1], "o-"
        )
        self.figure.show()

        n = self.xy.shape[0]
        numEl = np.sum(x for x in range(n))
        cities_ = np.linspace(0, n - 1, n).astype(int)
        cities = ["" for i in range(cities_.size)]
        for i in range(cities_.size):
            cities[i] = np.array2string(cities_[i])

        edges = []
        totalDist = 0
        dist_dict = {c: {} for c in cities}
        for idx_1 in range(0, len(cities) - 1):
            for idx_2 in range(idx_1 + 1, len(cities)):
                city_a = cities[idx_1]  # xy[idx_1,0].astype(int)
                city_b = cities[idx_2]  # xy[idx_2,0].astype(int)
                # dist = get_distance(xy[int(city_a), 0], xy[int(city_a), 1], xy[int(city_b), 0], xy[int(city_b), 1], )
                dist = np.sqrt(
                    (self.xy[int(city_a), 0] - self.xy[int(city_b), 0]) ** 2
                    + (self.xy[int(city_a), 1] - self.xy[int(city_b), 1]) ** 2
                )
                totalDist += dist
                dist_dict[city_a][city_b] = dist
                edges.append((city_a, city_b, dist))

        self.meanDist = totalDist / numEl
        self.hash_map = dist_dict
        self.cities = [k for k in self.hash_map.keys()]
        self.cities.remove(start)
        self.genes = []
        self.generate_genes = vectorize(self.generate_genes)
        self.generate_genes()

    def generate_genes(self):
        for i in range(self.population_size):
            gene = [self.start]
            city_a = "0"
            options = [k for k in self.cities]
            loop = 0
            threshold = len(self.cities)
            while len(gene) < len(self.cities) + 1:
                city_b = random.choice(options)
                try:
                    dist = self.hash_map[city_a][city_b]
                except:
                    dist = self.hash_map[city_b][city_a]
                if (dist > self.meanDist / 4) & (loop < threshold):
                    loop += 1
                    continue
                if (dist > self.meanDist / 3) & (loop < 2 * threshold):
                    loop += 1
                    continue
                if (dist > self.meanDist / 2) & (loop < 3 * threshold):
                    loop += 1
                    continue
                if (dist > self.meanDist) & (loop < 4 * threshold):
                    loop += 1
                    continue
                loc = options.index(city_b)
                loop = 0
                gene.append(city_b)
                del options[loc]
                city_a = city_b
            gene.append(self.start)
            self.genes.append(gene)
        # return self.genes

    def GA_Matlab(self):
        globalMin = np.Inf
        distHistory = np.zeros((self.iterations, 1))
        tmpPop = []
        for i in range(4):
            tmpPop.append(self.genes[i])
        newPop = []

        counter = 0
        for iter in range(self.iterations):
            counter = counter + 1
            totalDist = []
            # Evaluate Each Population Member(Calculate Total Distance)
            for geneP in self.genes:  # range(self.population_size):
                d = 0  # self.hash_map[geneP[-1]][geneP[0]] # Closed Path
                for k in range(1, len(geneP)):
                    try:
                        city_a = geneP[k - 1]
                        city_b = geneP[k]
                        dist = self.hash_map[city_a][city_b]
                    except:
                        dist = self.hash_map[city_b][city_a]
                    d += dist
                totalDist.append(d)

            #  Find the Best Route in the Population
            if iter == 0:
                minDist = totalDist[0]
                index = 0
                distHistory[iter] = minDist
            else:
                minDist = np.amin(totalDist)
                index = np.argmin(totalDist)
                distHistory[iter] = minDist
            if minDist < globalMin:
                globalMin = minDist
                optRoute = self.genes[index]
            del newPop
            newPop = []
            # Genetic Algorithm Operators
            randomOrder = np.random.permutation(self.population_size)
            for p in range(4, self.population_size + 1, 4):
                randomOrderP = randomOrder[p - 4 : p]
                rtes = []
                dists = []
                for ii in range(4):
                    randomOrderP2 = randomOrderP[ii]
                    rtes.append(self.genes[randomOrderP2])
                    dists.append(totalDist[randomOrderP2])
                idx = np.argmin(dists)
                bestOf4Route = rtes[idx]
                routeInsertionPoints = np.transpose(
                    np.sort(np.ceil((len(geneP) - 2) * np.random.rand(1, 2)))
                )
                I = int(routeInsertionPoints[0])
                J = int(routeInsertionPoints[1])
                for k in range(4):  # Mutate the Best to get Three New Routes
                    tmpPop[k] = bestOf4Route.copy()
                    if k == 1:  # Flip
                        index_a = []
                        for ii in range(I, J + 1):
                            index_a.append(tmpPop[k][J + I - ii])
                        jj = 0
                        for ii in range(I, J + 1):
                            tmpPop[k][ii] = index_a[jj]
                            jj += 1
                    elif k == 2:  # Swap
                        index_a = tmpPop[k][I]
                        index_b = tmpPop[k][J]
                        tmpPop[k][I] = index_b
                        tmpPop[k][J] = index_a
                    elif k == 3:  # Slide
                        index_a = []
                        for ii in range(I + 1, J + 1):
                            index_a.append(tmpPop[k][ii])
                        index_b = tmpPop[k][I]
                        jj = 0
                        for ii in range(I, J):
                            tmpPop[k][ii] = index_a[jj]
                            jj += 1
                        tmpPop[k][J] = index_b
                    newPop.append(tmpPop[k])
            if iter % (self.iterations // self.plotUpdateFrequency) == 0:
                print("distance: %i um, numIteration: %i, " % (globalMin, iter))
                current_best_gene_r = np.array(optRoute).astype(int)

                self.ax_scanGridOpt_plot.set_data(
                    self.xy[current_best_gene_r, 0], self.xy[current_best_gene_r, 1]
                )
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()

            del self.genes
            self.genes = newPop.copy()
        return globalMin, optRoute

    def converge(self):
        values = self.GA_Matlab()
        current_score = values[0]
        current_best_gene = values[1]
        return np.array(current_best_gene).astype(int)
