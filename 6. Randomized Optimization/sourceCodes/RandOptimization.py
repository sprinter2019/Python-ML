## MAchine Learning - Assignment 1
## Georgia Institute of Technology
## Applying ML supervised algorithms to interesting datasets

import matplotlib.pyplot as plt
import sys
import psutil
import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count

import sklearn
import mlrose



maxIter=1000000000
maxAttempts=50

## Fitness Function
coords_list= [(2,10), (5,2), (6,11), (7,4), (15,4), (4,6), (2,5), (3,3), (10,1), 
              (4,2), (9,2), (6,4), (14,24), (3,6), (1,5), (2,3), (1,2), (4,3), 
              (6,3), (6,5), (4,5), (13,7), (2,8), (2,4)]


### Knapsack
weights=[1, 5, 2, 4, 1, 1, 5, 2, 4, 1, 3, 1, 3, 2, 3, 3, 1, 3, 2, 3]
values= [1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10, 1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10]
max_weight_pct=0.6
 
fitness=mlrose.Knapsack(weights, values, max_weight_pct)
 
state=np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
 
problem=mlrose.DiscreteOpt(length=20, fitness_fn=fitness, maximize=True);
 
best_state_ga, best_fitness_ga, best_curve_ga=mlrose.genetic_alg(problem, max_iters=maxIter, curve=True, max_attempts=maxAttempts)

print(best_state_ga)
print(best_fitness_ga)
#print(best_curve_ga)

best_state_rhc, best_fitness_rhc, best_curve_rhc=mlrose.random_hill_climb(problem, max_iters=maxIter, curve=True, max_attempts=maxAttempts)
#print(best_state_rhc)
print(best_fitness_rhc)
#print(best_curve_rhc)

best_state_sa, best_fitness_sa, best_curve_sa=mlrose.simulated_annealing(problem, max_iters=maxIter, curve=True, max_attempts=maxAttempts)

#print(best_state_sa)
print(best_fitness_sa)
#print(best_curve_sa)

best_state_mm, best_fitness_mm, best_curve_mm=mlrose.mimic(problem, max_iters=maxIter, curve=True, max_attempts=maxAttempts)

#print(best_state_mm)
print(best_fitness_mm)
#print(best_curve_mm)


###TSP
'''
fitness= mlrose.TravellingSales(coords= coords_list);

## Optimization Function
problem=mlrose.TSPOpt(length=24, fitness_fn=fitness, maximize=True)

## Optimization Algorithm
_, best_fitness_ga=mlrose.genetic_alg(problem, max_attempts=100)

#print(best_state_ga)
print(best_fitness_ga)

_, best_fitness_rhc=mlrose.random_hill_climb(problem, max_attempts=100)

print(best_fitness_rhc)

_, best_fitness_sa=mlrose.simulated_annealing(problem, max_attempts=100)

print(best_fitness_sa)

_, best_fitness_mm=mlrose.mimic(problem, max_attempts=100)

print(best_fitness_mm)
'''


##def geneticAlgorithm (problem):

## mlrose.random_hill_climb(problem, max_attempts=20)


## mlrose.simulated_annealing(problem, max_attempts=20)


## mlrose.genetic_alg(problem, max_attempts=20)


## mlrose.mimic(problem, max_attempts=20)