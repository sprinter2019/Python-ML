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


# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
   # Initialize counter
   fitness_cnt = 0
   # For all pairs of queens
   for i in range(len(state) - 1):
       for j in range(i + 1, len(state)):
           # Check for horizontal, diagonal-up and diagonal-down attacks
           if (state[j] != state[i]) \
               and (state[j] != state[i] + (j - i)) \
               and (state[j] != state[i] - (j - i)):
                   # If no attacks, then increment counter
                   fitness_cnt += 1

   return fitness_cnt

# Initialize custom fitness function object


init_state=np.array([0, 1, 2, 3, 4, 5, 6, 7])

##fitness=mlrose.MaxKColor(edges);
fitness = mlrose.CustomFitness(queens_max)


problem=mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = True, max_val = 8);

maxIter=[10, 100, 500, 1000, 2000, 3000, 5000, 7000, 10000]
##maxIter=[10, 100]
maxAttempts=100
popSize=1000
muProb=0.1


print("Simulating Genetic Algorithm.......")
for i in range(len(maxIter)):
    if i==0:
        scoreFitnessGA=[]
        usageTimeGA=[]
        
    start_time = time.time()
    best_state_ga, best_fitness_ga, best_curve_ga=mlrose.genetic_alg(problem, max_iters=maxIter[i], 
                            pop_size=popSize, mutation_prob=muProb, curve=True, max_attempts=maxAttempts, random_state = 10)


    usageTimeGA.append(time.time()-start_time);
    scoreFitnessGA.append(best_fitness_ga)
    
    
    
print("Simulating Random Hill Climbing Algorithm.......")
for i in range(len(maxIter)):
    if i==0:
        scoreFitnessRHC=[]
        usageTimeRHC=[]
    
        
    start_time = time.time()
    best_state_rhc, best_fitness_rhc, best_curve_rhc=mlrose.random_hill_climb(problem, max_iters=maxIter[i], 
                                                                curve=True, max_attempts=maxAttempts, random_state = 10)



    usageTimeRHC.append(time.time()-start_time);
    scoreFitnessRHC.append(best_fitness_rhc)
    

print("Simulating Simulated Annealing Algorithm.......")
for i in range(len(maxIter)):
    if i==0:
        scoreFitnessSA=[]
        usageTimeSA=[]
        
    
    start_time = time.time()    
    best_state_sa, best_fitness_sa, best_curve_sa=mlrose.simulated_annealing(problem, max_iters=maxIter[i], 
                                                                curve=True, max_attempts=maxAttempts, random_state = 10)




    usageTimeSA.append(time.time()-start_time);
    scoreFitnessSA.append(best_fitness_sa)
    
    
    
print("Simulating MIMIC Algorithm.......")
for i in range(len(maxIter)):
    if i==0:
        scoreFitnessMM=[]
        usageTimeMM=[]
        
    start_time = time.time()
    best_state_mm, best_fitness_mm, best_curve_mm=mlrose.mimic(problem, max_iters=maxIter[i], 
                                pop_size=popSize, keep_pct= 0.2, curve=True, max_attempts=maxAttempts, random_state = 10)



    usageTimeMM.append(time.time()-start_time);
    scoreFitnessMM.append(best_fitness_mm)    
    
  
    
    
    
plt.figure('FlipFlop_01')
plt.title('Fitness vs Iterations', fontsize=16);
plt.ylabel('Fitness', fontsize=14);
plt.xlabel('Number of Iterations', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(40, 65)
plt.plot(maxIter, scoreFitnessGA, 'go', label='Genetic Algorithm')
plt.plot(maxIter, scoreFitnessRHC, 'rs' , label='Random Hill Climbing')
plt.plot(maxIter, scoreFitnessSA, 'b*', label='Simulated Annealing')
plt.plot(maxIter, scoreFitnessMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/FlipFlop_Fitness.png')
plt.show(block=False)
    

plt.figure('FlipFlop_02')
plt.title('Run Time vs Iterations', fontsize=16);
plt.ylabel('Run Time (s)', fontsize=14);
plt.xlabel('Number of Iterations', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(40, 65)
plt.plot(maxIter, usageTimeGA, 'go', label='Genetic Algorithm')
plt.plot(maxIter, usageTimeRHC, 'rs' , label='Random Hill Climbing')
plt.plot(maxIter, usageTimeSA, 'b*', label='Simulated Annealing')
plt.plot(maxIter, usageTimeMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/FlipFlop_RunTime.png')
plt.show()