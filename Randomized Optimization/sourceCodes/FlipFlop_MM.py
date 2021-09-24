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



fitness=mlrose.FlipFlop();

problem=mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2);

maxIter=10000
##maxIter=[10, 100]
maxAttempts=10
#popSize=1000
muProb=0.1
keep_pct=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
    
    
print("Simulating MIMIC Algorithm.......")
for i in range(len(keep_pct)):
    if i==0:
        scoreFitnessMM=[]
        keepPct=[]
        usageTimeMM=[]
        
    start_time = time.time()
    best_state_mm, best_fitness_mm, best_curve_mm=mlrose.mimic(problem, max_iters=maxIter, 
                                 keep_pct= keep_pct[i], curve=True, max_attempts=maxAttempts, random_state = 3)



    usageTimeMM.append(time.time()-start_time);
    scoreFitnessMM.append(best_fitness_mm) 
    keepPct.append(keep_pct[i]*100) 
    
  
    
    
    
plt.figure('FlipFlop_01')
plt.title('Fitness vs Keep Percentage', fontsize=16);
plt.ylabel('Fitness', fontsize=14);
plt.xlabel('Keep Percentage (%)', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(40, 90)
#plt.plot(maxIter, scoreFitnessGA, 'go', label='Genetic Algorithm')
#plt.plot(maxIter, scoreFitnessRHC, 'rs' , label='Random Hill Climbing')
#plt.plot(maxIter, scoreFitnessSA, 'b*', label='Simulated Annealing')
plt.plot(keepPct, scoreFitnessMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/FlipFlop_Fitness_MM.png')
plt.show(block=False)
    

plt.figure('FlipFlop_02')
plt.title('Run Time vs Keep Percentage', fontsize=16);
plt.ylabel('Run Time (s)', fontsize=14);
plt.xlabel('Keep Percentage (%)', fontsize=14);
#plt.xlim(0, 100)
#plt.ylim(40, 65)
plt.yscale('log')
#plt.plot(maxIter, usageTimeGA, 'go', label='Genetic Algorithm')
#plt.plot(maxIter, usageTimeRHC, 'rs' , label='Random Hill Climbing')
#plt.plot(maxIter, usageTimeSA, 'b*', label='Simulated Annealing')
plt.plot(keepPct, usageTimeMM, 'mD' , label='MIMIC')
plt.legend(loc='best', fontsize=12)
plt.savefig('./Plots/FlipFlop_RunTime_MM.png')
plt.show()