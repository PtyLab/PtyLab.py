import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from fracPy.utils.scanGrids import GenerateNonUniformFermat
import matplotlib.pylab as plt

# def get_distance(start, stop):
#     api = "getYourOwnKeyDude"
#     url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins=" + start + "&destinations=" + stop + "&key=" + api
#     link = requests.get(url)
#     json_loc = link.json()
#     distance = json_loc['rows'][0]['elements'][0]['distance']['text']
#     distance = int(''.join([d for d in distance if d.isdigit() == True]))
#     return distance


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

    def __init__(self, hash_map, xy, meanDist, start, steps=2, crossover_prob=0.15, mutation_prob=0.15, population_size=5,
                 iterations=100):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population_size = population_size
        self.hash_map = hash_map
        self.xy = xy
        self.steps = steps
        self.iterations = iterations
        self.start = start
        self.meanDist = meanDist
        self.cities = [k for k in self.hash_map.keys()]
        self.cities.remove(start)
        self.genes = []
        self.epsilon = 1 - 1 / self.iterations
        self.generate_genes = vectorize(self.generate_genes)
        self.evaluate_fitness = vectorize(self.evaluate_fitness)
        self.evolve = vectorize(self.evolve)
        self.prune_genes = vectorize(self.prune_genes)
        self.converge = vectorize(self.converge)

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

    def evaluate_fitness(self):
        fitness_scores = []
        for gene in self.genes:
            total_distance = 0
            for idx in range(1, len(gene)):
                city_b = gene[idx]
                city_a = gene[idx - 1]
                try:
                    dist = self.hash_map[city_a][city_b]
                except:
                    dist = self.hash_map[city_b][city_a]
                total_distance += dist
            fitness = 1 / total_distance
            fitness_scores.append(fitness)
        return fitness_scores

    def evolve(self):
        # index_map = {i: '' for i in range(len(self.cities))}
        index_map = {i: '' for i in range(1, len(self.cities) - 1)}
        # indices = [i for i in range(len(self.cities))]
        indices = [i for i in range(1, len(self.cities) - 1)]
        to_visit = [c for c in self.cities]
        cross = (1 - self.epsilon) * self.crossover_prob
        mutate = self.epsilon * self.mutation_prob
        crossed_count = int(cross * len(self.cities) - 1)
        mutated_count = int((mutate * len(self.cities) - 1) / 2)
        for idx in range(len(self.genes) - 1):
            gene = self.genes[idx]
            for i in range(crossed_count):
                try:
                    gene_index = random.choice(indices)
                    sample = gene[gene_index]
                    if sample in to_visit:
                        index_map[gene_index] = sample
                        loc = indices.index(gene_index)
                        del indices[loc]
                        loc = to_visit.index(sample)
                        del to_visit[loc]
                    else:
                        continue
                except:
                    pass
        indexRand = random.randint(0, self.population_size - self.steps - 1)
        last_gene = self.genes[indexRand]
        remaining_cities = [c for c in last_gene if c in to_visit]
        for k, v in index_map.items():
            if v != '':
                continue
            else:
                city = remaining_cities.pop(0)
                index_map[k] = city
        # new_gene = [index_map[i] for i in range(len(self.cities))]
        new_gene = [index_map[i] for i in range(1, len(self.cities) - 1)]
        new_gene.insert(0, self.start)
        new_gene.append(self.start)
        for i in range(mutated_count):
            choices2 = []
            flag = False
            choices1 = [c for c in new_gene if c != self.start]
            for jj in range(1, len(self.cities)-1):
                try:
                    city_a = new_gene[jj]
                    city_b = new_gene[jj-1]
                    dist = self.hash_map[city_a][city_b]
                except:
                    dist = self.hash_map[city_b][city_a]
                if dist > self.meanDist/1.5*self.epsilon:
                    flag = True
                    choices2.append(new_gene[jj])
                    if jj > 1:
                        choices2.append(new_gene[jj-1])
            # choices2 = faulty_nodes(new_gene)
            if flag == False:
                choices2 = choices1
            if len(choices2) > 2:
                choices1 = choices2
            city_a = random.choice(choices1)
            city_b = random.choice(choices2)
            index_a = new_gene.index(city_a)
            index_b = new_gene.index(city_b)
            new_gene[index_a] = city_b
            new_gene[index_b] = city_a
        self.genes.append(new_gene)

    # def faulty_nodes(gene):
    #     for c in gene if c != self.start:
    #         if c
    def GA_Matlab(self):
        globalMin = np.Inf
        totalDist = np.zeros((1, self.population_size))
        distHistory = np.zeros((self.iterations,1))
        tmpPop = []
        for i in range(4):
            tmpPop.append(self.genes[i])
        newPop = []  # self.genes.copy()

        counter = 0
        for iter in range(self.iterations):
            counter = counter + 1
            totalDist = []
        # Evaluate Each Population Member(Calculate Total Distance)
            for geneP in self.genes: #range(self.population_size):
                # geneP = self.genes[p]
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
                # rtes = pop(randomOrder(p - 3:p),:);
                # dists = totalDist(randomOrder(p - 3:p));
                idx = np.argmin(dists)  # ok
                bestOf4Route = rtes[idx]
                # % routeInsertionPoints = sort(ceil(n * rand(1, 2)));
                routeInsertionPoints = np.transpose(np.sort(np.ceil((len(geneP) - 2) * np.random.rand(1, 2))))
                I = int(routeInsertionPoints[0])
                J = int(routeInsertionPoints[1])
                for k in range(4): # Mutate the Best to get Three New Routes
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

    def prune_genes(self):
        values = self.GA_Matlab()
        # fitness_scores = self.evaluate_fitness()
        # for i in range(self.steps):
        #     worst_gene_index = fitness_scores.index(min(fitness_scores))
        #     del self.genes[worst_gene_index]
        #     del fitness_scores[worst_gene_index]
        # for i in range(self.steps):
        #     self.evolve()
        # fitness_scores = self.evaluate_fitness()
        # return max(fitness_scores), self.genes[fitness_scores.index(max(fitness_scores))]
        return values[0], values[1]

    def converge(self):
        for i in range(1):
            values = self.prune_genes()
            current_score = values[0]
            current_best_gene = values[1]
            self.epsilon -= 1 / self.iterations
            # if i % 10 == 0:
            #     print(f"{int(1 / current_score)} um")
            # if i % 200 == 0:
            #     current_best_gene_r = np.array(current_best_gene).astype(int)
            #     Rnew = np.transpose(xy[current_best_gene_r, 1])
            #     Cnew = np.transpose(xy[current_best_gene_r, 0])
            #     # show scan grid
            #     plt.figure(figsize=(5, 5), num=99)
            #     plt.plot(Rnew, Cnew, 'o-')
            #     plt.xlabel('um')
            #     plt.title('scan grid')
            #     plt.show(block=False)

        return current_best_gene


g = GeneticAlgo(hash_map=dist_dict, xy = xy, meanDist = meanDist, start='0', mutation_prob=0.25, crossover_prob=0.25,
                population_size=80, steps=50, iterations=500000)
current_best_gene_l = g.converge()
current_best_gene = np.array(current_best_gene_l).astype(int)
Rnew = np.transpose(xy[current_best_gene,1])
Cnew = np.transpose(xy[current_best_gene,0])
# show scan grid
# plt.figure(figsize=(5, 5), num=99)
# plt.plot(Rnew, Cnew, 'o-')
# plt.xlabel('um')
# plt.title('scan grid')
# plt.show(block=False)
