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


fitness= mlrose.TravellingSales(coords= coordsSupershargersCA);
#fitness= mlrose.TravellingSales(distances= dist_list);


problem=mlrose.TSPOpt(length=19, fitness_fn=fitness, maximize=False)

#problem=mlrose.TSPOpt(length=8, fitness_fn=fitness, maximize=True)



maxIter=10000
maxAttempts=10
popSize=[10, 100, 500, 1000, 2000, 3000, 4000, 5000]
muProb=0.1

        
print("Simulating Genetic Algorithm.......")
for i in range(len(popSize)):
    if i==0:
        scoreFitnessGA=[]
        usageTimeGA=[]
        
    start_time = time.time()
    best_state_ga, best_fitness_ga, best_curve_ga=mlrose.genetic_alg(problem, max_iters=maxIter, 
                            pop_size=popSize[i], mutation_prob=muProb, curve=True, max_attempts=maxAttempts, random_state = 3)


    usageTimeGA.append(time.time()-start_time);
    scoreFitnessGA.append(1/best_fitness_ga)
    
   
  
    
  
    
    
plt.figure('TWTesla_01')
plt.title('Fitness vs Population Size', fontsize=16);
plt.ylabel('Fitness', fontsize=14);
plt.xlabel('Population Size', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(0.003, 0.0035)
plt.plot(popSize, scoreFitnessGA, 'go', label='Genetic Algorithm')
#plt.plot(maxIter, scoreFitnessRHC, 'rs' , label='Random Hill Climbing')
#plt.plot(maxIter, scoreFitnessSA, 'b*', label='Simulated Annealing')
#plt.plot(maxIter, scoreFitnessMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/TWTesla_Fitness_GA.png')
plt.show(block=False)
    

plt.figure('TWTesla_02')
plt.title('Run Time vs Population Size', fontsize=16);
plt.ylabel('Run Time (s)', fontsize=14);
plt.xlabel('Population Size', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(-1, 20)
plt.yscale('log')
plt.plot(popSize, usageTimeGA, 'go', label='Genetic Algorithm')
#plt.plot(maxIter, usageTimeRHC, 'rs' , label='Random Hill Climbing')
#plt.plot(maxIter, usageTimeSA, 'b*', label='Simulated Annealing')
#plt.plot(maxIter, usageTimeMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/TWTesla_RunTime_GA.png')
plt.show()