import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from fracPy.utils.scanGrids import GenerateNonUniformFermat


def get_distance(start_x, start_y, stop_x, stop_y):
    distance = ((start_x-stop_x)**2 + (start_y-stop_y)**2)**(0.5)
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

import random
import operator


class GeneticAlgo():

    def __init__(self, xy, start_po, steps=1, crossover_prob=0.15, mutation_prob=0.15, population_size=40,
                 iterations=100):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population_size = population_size
        self.xy = xy
        self.steps = steps
        self.iterations = iterations
        self.start_po = start_po
        self.genes = []
        self.epsilon = 1 - 1 / self.iterations
        self.generate_genes = np.vectorize(self.generate_genes)
        self.evaluate_fitness = np.vectorize(self.evaluate_fitness)
        self.evolve = np.vectorize(self.evolve)
        self.prune_genes = np.vectorize(self.prune_genes)
        self.converge = np.vectorize(self.converge)

        self.generate_genes()

    def generate_genes(self):
        for i in range(self.population_size):
            gene = [self.start_po]
            options = [k for k in range(self.xy.shape[0])]
            while len(gene) < self.xy.shape[0] + 1:
                option_rand = random.choice(options)
                loc = options.index(option_rand)
                gene.append(option_rand)
                del options[loc]
            gene.append(self.start_po)
            self.genes.append(gene)
        return self.genes

    def evaluate_fitness(self):
        fitness_scores = []
        for gene in self.genes:
            total_distance = 0
            for idx in range(1, len(gene)):
                city_b = gene[idx]
                city_a = gene[idx - 1]
                dist = ((xy[city_b,0]-xy[city_a,0])**2 + (xy[city_b,1]-xy[city_a,1])**2)**(1/2)
                total_distance += dist
            total_distance += ((xy[-1,0]-xy[0,0])**2 + (xy[-1,1]-xy[0,1])**2)**(1/2)
            fitness = 1 / total_distance
            fitness_scores.append(fitness)
        return fitness_scores

    def evolve(self):
        index_map = {i: '' for i in range(1, self.xy.shape[0] - 1)}
        indices = [i for i in range(1, self.xy.shape[0] - 1)]
        to_visit = [c for c in range(1, self.xy.shape[0])]
        cross = (1 - self.epsilon) * self.crossover_prob
        mutate = self.epsilon * self.mutation_prob
        crossed_count = int(cross * self.xy.shape[0] - 1)
        mutated_count = int((mutate * self.xy.shape[0] - 1) / 2)
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
        last_gene = self.genes[-1]
        remaining_cities = [c for c in last_gene if c in to_visit]
        for k, v in index_map.items():
            if v != '':
                continue
            else:
                city = remaining_cities.pop(0)
                index_map[k] = city
        new_gene = [index_map[i] for i in range(1, self.xy.shape[0] - 1)]
        new_gene.insert(0, self.start_po)
        new_gene.append(self.start_po)
        for i in range(mutated_count):
            choices = [c for c in new_gene if c != self.start_po]
            city_a = random.choice(choices)
            city_b = random.choice(choices)
            index_a = new_gene.index(city_a)
            index_b = new_gene.index(city_b)
            new_gene[index_a] = city_b
            new_gene[index_b] = city_a
        self.genes.append(new_gene)

    def prune_genes(self):
        for i in range(self.steps):
            self.evolve()
        fitness_scores = self.evaluate_fitness()
        for i in range(self.steps):
            worst_gene_index = fitness_scores.index(min(fitness_scores))
            del self.genes[worst_gene_index]
            del fitness_scores[worst_gene_index]
        return max(fitness_scores), self.genes[fitness_scores.index(max(fitness_scores))]

    def converge(self):
        for i in range(self.iterations):
            values = self.prune_genes()
            current_score = values[0]
            current_best_gene = values[1]
            self.epsilon -= 1 / self.iterations
            if i % 100 == 0:
                print(f"{int(1 / current_score)} miles")
                plt.figure(figsize=(5, 5), num=99)
                plt.plot(R, C, 'o-')
                plt.xlabel('um')
                plt.title('scan grid')
                plt.show(block=False)
        return current_best_gene


g = GeneticAlgo(xy = xy, start_po=0, mutation_prob=0.25, crossover_prob=0.25,
                population_size=30, steps=15, iterations=2000)
g.converge()