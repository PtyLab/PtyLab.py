import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from fracPy.utils.scanGrids import GenerateNonUniformFermat
import matplotlib.pylab as plt

def get_distance(start_x, start_y, stop_x, stop_y):
    distance = ((start_x-stop_x)**2 + (start_y-stop_y)**2)**(1/2)
    return distance

def get_xy(mode, n):
    if mode == 'GenerateNonUniformFermat':
        R, C = GenerateNonUniformFermat(n)
    elif mode == 'GenerateConcentricGrid':
        R, C = GenerateConcentricGrid(n, 15, 3)
    elif mode == 'GenerateRasterGrid':
        R, C = GenerateRasterGrid(n, 5)
    xy = np.zeros((n,2))
    xy[:,0] = np.transpose(C)
    xy[:,1] = np.transpose(R)
    return xy

mode = 'GenerateNonUniformFermat'
n = 150
xy = get_xy(mode,n)
numEl = np.sum(x for x in range(n))
cities_ = np.linspace(0,n-1,n).astype(int)
cities = ["" for i in range(cities_.size)]
for i in range(cities_.size):
    cities[i] = np.array2string(cities_[i])
# xy[:,0] = np.transpose(cities)
# cities = [c for c in cities.split('\n') if c != '']

edges = []
totalDist = 0
dist_dict = {c: {} for c in cities}
for idx_1 in range(0, len(cities) - 1):
    for idx_2 in range(idx_1 + 1, len(cities)):
        city_a = cities[idx_1] #xy[idx_1,0].astype(int)
        city_b = cities[idx_2] #xy[idx_2,0].astype(int)
        dist = get_distance(xy[int(city_a),0],xy[int(city_a),1], xy[int(city_b),0],xy[int(city_b),1],)
        totalDist += dist
        dist_dict[city_a][city_b] = dist
        edges.append((city_a, city_b, dist))

meanDist = totalDist/numEl
import random
import operator
from numpy import vectorize


class GeneticAlgo():

    def __init__(self, hash_map, xy, meanDist, start, population_size=5,
                 iterations=100):
        self.population_size = population_size
        self.hash_map = hash_map
        self.xy = xy
        self.iterations = iterations
        self.start = start
        self.meanDist = meanDist
        self.cities = [k for k in self.hash_map.keys()]
        self.cities.remove(start)
        self.genes = []
        self.generate_genes = vectorize(self.generate_genes)

        self.generate_genes()

    def generate_genes(self):
        for i in range(self.population_size):
            gene = [self.start]
            city_a = '0'
            options = [k for k in self.cities]
            loop = 0
            threshold = len(self.cities)
            while len(gene) < len(self.cities) + 1:
                city_b = random.choice(options)
                try:
                    dist = self.hash_map[city_a][city_b]
                except:
                    dist = self.hash_map[city_b][city_a]
                if (dist > self.meanDist/4) & (loop < threshold):
                    loop += 1
                    continue
                if (dist > self.meanDist/3) & (loop < 2*threshold):
                    loop += 1
                    continue
                if (dist > self.meanDist/2) & (loop < 3*threshold):
                    loop += 1
                    continue
                if (dist > self.meanDist) & (loop < 4*threshold):
                    loop += 1
                    continue
                loc = options.index(city_b)
                loop = 0
                gene.append(city_b)
                del options[loc]
                city_a = city_b
            gene.append(self.start)
            self.genes.append(gene)
        return self.genes

    def GA_Matlab(self):
        globalMin = np.Inf
        distHistory = np.zeros((self.iterations,1))
        tmpPop = []
        for i in range(4):
            tmpPop.append(self.genes[i])
        newPop = []

        counter = 0
        for iter in range(self.iterations):
            counter = counter + 1
            totalDist = []
        # Evaluate Each Population Member(Calculate Total Distance)
            for geneP in self.genes: #range(self.population_size):
                d = 0 #self.hash_map[geneP[-1]][geneP[0]] # Closed Path
                for k in range(1,len(geneP)):
                    try:
                        city_a = geneP[k-1]
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
            for p in range(4, self.population_size+1,4):
                randomOrderP = randomOrder[p-4:p]
                rtes = []
                dists = []
                for ii in range(4):
                    randomOrderP2 = randomOrderP[ii]
                    rtes.append(self.genes[randomOrderP2])
                    dists.append(totalDist[randomOrderP2])
                idx = np.argmin(dists)
                bestOf4Route = rtes[idx]
                routeInsertionPoints = np.transpose(np.sort(np.ceil((len(geneP) - 2) * np.random.rand(1, 2))))
                I = int(routeInsertionPoints[0])
                J = int(routeInsertionPoints[1])
                for k in range(4):  # Mutate the Best to get Three New Routes
                    tmpPop[k] = bestOf4Route.copy()
                    if k == 1:  # Flip
                        index_a = []
                        for ii in range(I,J+1):
                            index_a.append(tmpPop[k][J+I-ii])
                        jj = 0
                        for ii in range(I,J+1):
                            tmpPop[k][ii] = index_a[jj]
                            jj += 1
                    elif k == 2:  # Swap
                        index_a = tmpPop[k][I]
                        index_b = tmpPop[k][J]
                        tmpPop[k][I] = index_b
                        tmpPop[k][J] = index_a
                    elif k == 3:  # Slide
                        index_a = []
                        for ii in range(I+1,J+1):
                            index_a.append(tmpPop[k][ii])
                        index_b = tmpPop[k][I]
                        jj = 0
                        for ii in range(I , J):
                            tmpPop[k][ii] = index_a[jj]
                            jj += 1
                        tmpPop[k][J] = index_b
                    newPop.append(tmpPop[k])
            if iter % 200 == 0:
                print(f"{int(globalMin)} um")
            if iter % 1000 == 0:
                current_best_gene_r = np.array(optRoute).astype(int)
                Rnew = np.transpose(self.xy[current_best_gene_r, 1])
                Cnew = np.transpose(self.xy[current_best_gene_r, 0])
                # show scan grid
                plt.figure(figsize=(5, 5), num=99)
                plt.plot(Rnew, Cnew, 'o-')
                plt.xlabel('um')
                plt.title('scan grid')
                plt.show(block=False)

            del self.genes
            self.genes = newPop.copy()
        return globalMin, optRoute

    def converge(self):
        values = self.GA_Matlab()
        current_score = values[0]
        current_best_gene = values[1]
        return current_best_gene


g = GeneticAlgo(hash_map=dist_dict, xy = xy, meanDist = meanDist, start='0',
                population_size=40,  iterations=50000)
current_best_gene_l = g.converge()
current_best_gene = np.array(current_best_gene_l).astype(int)
Rnew = np.transpose(xy[current_best_gene,1])
Cnew = np.transpose(xy[current_best_gene,0])
# show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(Rnew, Cnew, 'o-')
plt.xlabel('um')
plt.title('scan grid')
plt.show(block=False)
