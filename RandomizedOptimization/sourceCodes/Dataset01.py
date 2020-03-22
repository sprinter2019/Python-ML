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

#import sklearn
import mlrose
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn import ensemble
from sklearn.svm import SVC
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize



# ######################################## Neural Network (MLP Classifier) Analysis ############################################

def nnMlpAnalysis (x_data, y_data):
    print("Starting Neural Network Analysis With Back-Propagation.......", end = '');
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    trainAccuracy=[]
    testAccuracy=[]
    
    maxIter=[1, 10, 100, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    #maxIter=[1, 2, 5]

    #treeDepth=[]
    
    # XXX
    # TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
    # TODO: Create a SVC classifier and train it.
    # XXX
    mlp_tuned=MLPClassifier(activation='tanh', alpha=1e-07, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=True, epsilon=1e-08,
              hidden_layer_sizes=(250,), learning_rate='adaptive',
              learning_rate_init=0.001, max_iter=200, momentum=0.9,
              n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
              random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
              validation_fraction=0.1, verbose=False, warm_start=False)
    
    score = cross_val_score(mlp_tuned, x_test, y_test, cv=10)
    cvScore=score.mean()*100
        
    start_time = time.time()
    mlp_tuned.fit(x_train, y_train)
    fit_time=time.time()-start_time
    
    # XXX
    # TODO: Test its accuracy on the training set using the accuracy_score method.
    # TODO: Test its accuracy on the test set using the accuracy_score method.
    # XXX
    
    y_predict_test= mlp_tuned.predict(x_test)
    y_predict_train= mlp_tuned.predict(x_train)
    
    train=accuracy_score(y_train,y_predict_train.round())
    test=accuracy_score(y_test,y_predict_test.round())
    
    trainAccuracy.append(train*100)
    testAccuracy.append(test*100)
    #treeDepth.append(i);
    print("Done")    
    print("trainAccuracy:", trainAccuracy)
    print("CVAccuracy:", cvScore)
    print("testAccuracy:", testAccuracy)
    print("Fit Time:", fit_time)

    
#    print("End of Neural Network Analysis"); 
    
    
    print("Simulating Genetic Algorithm.......", end = '')
    for i in range(len(maxIter)):
        if i==0:
            trainAccuracyGA=[]
            testAccuracyGA=[]
            fitTimeGA=[]
          
        randOpt=mlrose.NeuralNetwork(hidden_nodes=[250], activation='tanh', algorithm='genetic_alg', max_iters=maxIter[i], bias=True,
                                 is_classifier= True, learning_rate=0.1, early_stopping=True, clip_max= 5,
                                 max_attempts=10, pop_size=1000, mutation_prob=0.1, random_state=100)
        
        start_time = time.time()    
        randOpt.fit(x_train, y_train)   
        fitTimeGA.append(time.time()-start_time);
        
        y_predict_test=randOpt.predict(x_test)
        y_predict_train=randOpt.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracyGA.append(train*100)
        testAccuracyGA.append(test*100)
        
    print("Done")   
    print("Simulating Random Hill Climbing Algorithm.......", end = '')
    for i in range(len(maxIter)):
        if i==0:
            trainAccuracyRHC=[]
            testAccuracyRHC=[]
            fitTimeRHC=[]
          
        randOpt=mlrose.NeuralNetwork(hidden_nodes=[250], activation='tanh', algorithm='random_hill_climb', max_iters=maxIter[i], bias=True,
                                 is_classifier= True, learning_rate=0.1, early_stopping=True, clip_max= 5,
                                 max_attempts=10, random_state=100)
        
        start_time = time.time()    
        randOpt.fit(x_train, y_train)   
        fitTimeRHC.append(time.time()-start_time);
        
        y_predict_test=randOpt.predict(x_test)
        y_predict_train=randOpt.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracyRHC.append(train*100)
        testAccuracyRHC.append(test*100)
        
    print("Done")
    print("Simulating Simulated Annealing Algorithm.......", end = '')
    for i in range(len(maxIter)):
        if i==0:
            trainAccuracySA=[]
            testAccuracySA=[]
            fitTimeSA=[]
          
        randOpt=mlrose.NeuralNetwork(hidden_nodes=[250], activation='tanh', algorithm='simulated_annealing', max_iters=maxIter[i], bias=True,
                                 is_classifier= True, learning_rate=0.1, early_stopping=True, clip_max= 5,
                                 max_attempts=10, random_state=100)
        
        start_time = time.time()    
        randOpt.fit(x_train, y_train)   
        fitTimeSA.append(time.time()-start_time);
        
        y_predict_test=randOpt.predict(x_test)
        y_predict_train=randOpt.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracySA.append(train*100)
        testAccuracySA.append(test*100)
    
    print("Done")
    
    ##### Print max accuracies
    print("SA_maxTrainAccuracy: ", max(trainAccuracySA)) 
    print("SA_maxTestAccuracy: ", max(testAccuracySA))
    print("RHC_maxTrainAccuracy: ", max(trainAccuracyRHC)) 
    print("RHC_maxTestAccuracy: ", max(testAccuracyRHC))
    print("GA_maxTrainAccuracy: ", max(trainAccuracyGA)) 
    print("GA_maxTestAccuracy: ", max(testAccuracyGA))
        
    #### Plots Section
    plt.figure('NN_01')
    plt.title('Accuracy vs Iterations', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Number of Iterations', fontsize=14);
    plt.xscale('log')
    #plt.xlim(0, 100)
    plt.ylim(45, 60)
    plt.plot(maxIter, trainAccuracyGA, 'g*:', label='GA_Train')
    plt.plot(maxIter, testAccuracyGA, 'gs-' , label='GA_Test')
    plt.plot(maxIter, trainAccuracySA, 'b*:', label='SA_Train')
    plt.plot(maxIter, testAccuracySA, 'bs-', label='SA_Test')
    plt.plot(maxIter, trainAccuracyRHC, 'r*:' , label='RHC_Train')
    plt.plot(maxIter, testAccuracyRHC, 'rs-', label='RHC_Test')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Plots/NN_opt_accuracy2.png')
    plt.show(block=False)
        
    
    plt.figure('NN_02')
    plt.title('Training Time vs Iterations', fontsize=16);
    plt.ylabel('Training Time (s)', fontsize=14);
    plt.xlabel('Number of Iterations', fontsize=14);
    #plt.xlim(0, 100)
    #plt.ylim(40, 65)
    plt.yscale('log')
    plt.plot(maxIter, fitTimeGA, 'go', label='Genetic Algorithm')
    plt.plot(maxIter, fitTimeRHC, 'rs' , label='Random Hill Climbing')
    plt.plot(maxIter, fitTimeSA, 'b*', label='Simulated Annealing')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Plots/NN_opt_fitTime2.png')
    plt.show()
      

###### End of Neural Network  Analysis ######

######################################### Reading and Splitting the Data ############################################

dataset=pd.read_csv('EEG_Eye_State.csv')

# Separate out the x_data and y_data.
x_data = dataset.loc[:, dataset.columns != "class"]
y_data = dataset.loc[:, "class"]

# TODO: Split 75% of the data into training and 25% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)

#nnMlpAnalysis (x_data, y_data)

nnMlpAnalysis (x_data, y_data) 