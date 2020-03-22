#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:25:36 2019

@author: khaled
"""

#### Travelling with a Tesla ####


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


maxIter=1000
maxAttempts=10

popSize=1000
muProb=0.1

## https://mortada.net/the-traveling-tesla-salesman.html
#https://teslamotorsclub.com/tmc/threads/a-list-of-all-supercharger-coordinates-and-a-script-to-grab-them.37948/

#### Corordinates of the superchargers in California, USA (2014) ####
coordsSupershargersCA= [(47.8803440000, 10.0403420000),
                        (35.4865850000, -120.6663780000),
                        (34.8491090000, -117.0854420000),
                        (34.6145550000, -120.1884320000),
                        (39.9264600000, -122.1984000000),
                        (38.6422910000, -121.1881300000),
                        (37.4924390000, -121.9447250000),
                        (37.0244500000, -121.5653500000),
                        (36.2541430000, -120.2379200000),
                        (33.9210630000, -118.3300740000),
                        (41.3102220000, -122.3173100000),
                        (38.7712080000, -121.2661490000),
                        (33.4985380000, -117.6630900000),
                        (34.9873700000, -118.9462720000),
                        (38.3666450000, -121.9581360000),
                        (39.3274380000, -120.2074100000),
                        (34.2381150000, -119.1780840000),
                        (33.7412910000, -116.2150290000),
                        (34.1135840000, -117.5294270000)]

coordsMayanmar=[(16.47, 96.10),
                (16.47, 94.44),
                (20.09, 92.54),
                (22.39, 93.37),
                (25.23, 97.24),
                (22.00, 96.05),
                (17.20, 96.29),
                (16.30, 97.38),
                (14.05, 98.12),
                (16.53, 97.38),
                (21.52, 95.59),
                (19.41, 97.13),
                (20.09, 94.55)]


coords_list= [(2,10), (5,2), (6,11), (7,4), (15,4), (4,6), (2,5), (3,3), (10,1), 
              (4,2), (9,2), (6,4), (14,23), (3,6), (1,5), (2,3), (1,2), (4,3), 
              (6,3), (6,5), (4,5), (13,7), (2,8), (2,4)]

# Create list of distances between pairs of cities
dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
(0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
(1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
(1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
(2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
(3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
(4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]


fitness= mlrose.TravellingSales(coords= coordsSupershargersCA);

##fitness= mlrose.TravellingSales(distances= dist_list);



problem=mlrose.TSPOpt(length=19, fitness_fn=fitness, maximize=False)

#problem=mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2);

best_state_ga, best_fitness_ga, best_curve_ga=mlrose.genetic_alg(problem, max_iters=maxIter, 
                        pop_size=popSize, mutation_prob=muProb, curve=True, max_attempts=maxAttempts, random_state = 3)

#print(best_state_ga)
print(best_fitness_ga)
#print(best_curve_ga)

best_state_rhc, best_fitness_rhc, best_curve_rhc=mlrose.random_hill_climb(problem, max_iters=maxIter, curve=True, 
                                                                          max_attempts=maxAttempts, random_state = 3)
#print(best_state_rhc)
print(best_fitness_rhc)
#print(best_curve_rhc)

best_state_sa, best_fitness_sa, best_curve_sa=mlrose.simulated_annealing(problem, max_iters=maxIter, curve=True, 
                                                                         max_attempts=maxAttempts, random_state = 3)

#print(best_state_sa)
print(best_fitness_sa)
#print(best_curve_sa)

best_state_mm, best_fitness_mm, best_curve_mm=mlrose.mimic(problem, max_iters=maxIter, curve=True, 
                                                           keep_pct= 0.2, max_attempts=maxAttempts, random_state = 3)

#print(best_state_mm)
print(best_fitness_mm)
#print(best_curve_mm)