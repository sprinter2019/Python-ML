#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 06:39:13 2019

@author: khaled
"""

#https://www.novatec-gmbh.de/en/blog/introduction-to-q-learning/

import gym.spaces
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns
import time
from statistics import mean 

 
#env  = gym.make('FrozenLake-v0')
env = gym.make("Taxi-v3")
state_space = env.observation_space.n
action_space = env.action_space.n
 
qtable = np.zeros((state_space, action_space))

epsilon = 1.0          #Greed 100%
epsilon_min = 0.005     #Minimum greed 0.05%
epsilon_decay = 0.99995 #Decay multiplied with epsilon after each episode
episodes = 100000        #Amount of games
max_steps = 1000        #Maximum steps per episode
learning_rate = 0.6
gamma = 0.9

allScores=[]
xEpisodes=[]
iters=[]
ctime=[]
for episode in range(episodes):
 
    # Reset the game state, done and score before every episode/game
    state = env.reset() #Gets current game state
    done = False        #decides whether the game is over
    score = 0
 
    start=time.time()
    for i in range(max_steps):
 
        # With the probability of (1 - epsilon) take the best action in our Q-table
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[state, :])
        # Else take a random action
        else:
            action = env.action_space.sample()
        
        # Step the game forward
        next_state, reward, done, _ = env.step(action)
 
        # Add up the score
        score += reward
 
        # Update our Q-table with our Q-function
        qtable[state, action] = (1 - learning_rate) * qtable[state, action] \
            + learning_rate * (reward + gamma * np.max(qtable[next_state,:]))
 
        # Set the next state as the current state
        state = next_state
 
        if done:
            ctime.append(time.time()-start)
            allScores.append(score)
            xEpisodes.append(episode)
            iters.append(i)
            break
 
    # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
    if epsilon >= epsilon_min:
        epsilon *= epsilon_decay


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = 1
chunks = list(chunk_list(allScores, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]

dir = os.path.join("./","A4_Plots")
if not os.path.exists(dir):
    os.mkdir(dir)


print("Maximum Iteration: ", max(iters))
print("Minimum Iteration: ", min(iters))
print("Average Iteration: ", mean(iters))
print("Maximum convergence time: ", max(ctime))
print("Minimum convergence time: ", min(ctime))
print("Average convergence time: ", mean(ctime))
print("Maximum Score: ", max(averages))
print("Minimum Score: ", min(averages))
print("Average Score: ", mean(averages))

#sns.distplot(averages)

 
#print(allScores)
#print(xEpisodes)     
#plt.plot(xEpisodes, allScores, 'go', label='Score')
plt.title('Score vs Number of Episodes', fontsize=16);
plt.plot(range(0, len(allScores), size), averages, 'go', label='Score')
plt.legend(loc='best', fontsize=12)

plt.xlabel('Episode')
plt.ylabel('Score')
#plt.xscale('log')
#plt.yscale('log')

plt.savefig('./A4_Plots/qTaxi_score_LR60_Eps10.png', bbox_inches='tight')
plt.show(block=False)


plt.figure()
plt.title('Number of Iterations vs Number of Episodes', fontsize=16);
plt.plot(range(0, len(allScores), size), iters, 'bo', label='Iterations')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Episode')
plt.ylabel('Iterations')
plt.savefig('./A4_Plots/qTaxi_iter_LR60_Eps10.png', bbox_inches='tight')
plt.show(block=False)



plt.figure()
plt.title('Convergence time vs Number of Episodes', fontsize=16);
plt.plot(range(0, len(allScores), size), ctime, 'ro', label='convergence time')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Episode')
plt.ylabel('Time (s)')
plt.savefig('./A4_Plots/qTaxi_time_LR60_Eps10.png', bbox_inches='tight')
plt.show()

#sns.distplot(iters)
