#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:25:36 2019

@author: khaled
"""

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


maxIter=10000
maxAttempts=50
popSize=1000
muProb=0.1


fitness=mlrose.OneMax();
problem=mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2);

best_state_ga, best_fitness_ga, best_curve_ga=mlrose.genetic_alg(problem, max_iters=maxIter, 
                        pop_size=popSize, mutation_prob=muProb, curve=True, max_attempts=maxAttempts)

#print(best_state_ga)
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

best_state_mm, best_fitness_mm, best_curve_mm=mlrose.mimic(problem, max_iters=maxIter, pop_size=popSize, keep_pct= muProb, curve=True, max_attempts=maxAttempts)

#print(best_state_mm)
print(best_fitness_mm)
#print(best_curve_mm)