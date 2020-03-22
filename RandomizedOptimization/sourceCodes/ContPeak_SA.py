#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:40:52 2019

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


#fitness=mlrose.ContinuousPeaks(t_pct=0.15);
#problem=mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=5);


maxIter=10000

decay=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
init_temp=[1, 5, 10, 15 , 20, 25]

exp_const=[0.005, 0.05, 0.1, 0.2, 0.5, 1]
t_pct=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.4, 0.7, 1.0]

#maxIter=[10, 100]
maxAttempts=10
popSize=1000
muProb=0.1
    
    

print("Simulating Simulated Annealing Algorithm.......")
for i in range(len(t_pct)):
    fitness=mlrose.ContinuousPeaks(t_pct=t_pct[i]);
    problem=mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=5);
    if i==0:
        scoreFitnessSA=[]
        usageTimeSA=[]
    #schedule=mlrose.GeomDecay()
    #schedule=mlrose.ExpDecay(exp_const=exp_const[i])
    start_time = time.time()    
    best_state_sa, best_fitness_sa, best_curve_sa=mlrose.simulated_annealing(problem, max_iters=maxIter,
                                                                curve=True, max_attempts=maxAttempts, random_state = 3)




    usageTimeSA.append(time.time()-start_time);
    scoreFitnessSA.append(best_fitness_sa)
    

    
  
    
    
    
plt.figure('ContPeak_01')
plt.title('Fitness vs Threshold Parameter', fontsize=16);
plt.ylabel('Fitness', fontsize=14);
plt.xlabel('Threshold Parameter', fontsize=14);

plt.xscale('log')
#plt.xlim(0, 100)
#plt.ylim(40, 65)
#plt.plot(maxIter, scoreFitnessGA, 'go', label='Genetic Algorithm')
#plt.plot(maxIter, scoreFitnessRHC, 'rs' , label='Random Hill Climbing')
plt.plot(t_pct, scoreFitnessSA, 'b*', label='Simulated Annealing')
#plt.plot(maxIter, scoreFitnessMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/ContPeak_Fitness_SA.png')
plt.show(block=False)
    

plt.figure('ContPeak_02')
plt.title('Run Time vs Threshold Parameter', fontsize=16);
plt.ylabel('Run Time (s)', fontsize=14);
plt.xlabel('Threshold Parameter', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(40, 65)
plt.yscale('log')
plt.xscale('log')
#plt.plot(maxIter, usageTimeGA, 'go', label='Genetic Algorithm')
#plt.plot(maxIter, usageTimeRHC, 'rs' , label='Random Hill Climbing')
plt.plot(t_pct, usageTimeSA, 'b*', label='Simulated Annealing')
#plt.plot(maxIter, usageTimeMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/ContPeak_RunTime_SA.png')
plt.show()
    
'''    
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
'''