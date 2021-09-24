## MAchine Learning - Assignment 1
## Georgia Institute of Technology
## Applying ML supervised algorithms to interesting datasets

import matplotlib.pyplot as plt
import sys
import psutil
import os

import numpy as np
from numpy import reshape
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, normalize, MinMaxScaler



# ######################################## Neural Network (MLP Classifier) Analysis ############################################

def nnMlpAnalysis (x_data, y_data, algo):
    print("Starting Neural Network Analysis: ", algo);
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    x_train, x_test, y_train_temp, y_test_temp = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    trainAccuracy=[]
    testAccuracy=[]
    #treeDepth=[]
    
    # XXX
    # TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
    # TODO: Create a SVC classifier and train it.
    # XXX
    x_data_norm = normalize(x_data, norm='l2')
    x_train_norm, x_test_norm, y_train_temp, y_test_temp = train_test_split(x_data_norm, y_data, test_size=0.25, random_state=100)


    one_hot=OneHotEncoder()
    
    y_train=one_hot.fit_transform(y_train_temp.values.reshape(-1,1)).todense()
    y_test=one_hot.fit_transform(y_test_temp.values.reshape(-1,1)).todense()

    
    
    
    #opt=MLPClassifier(learning_rate='adaptive', early_stopping=True, tol=1e-5, n_iter_no_change=25, random_state=100)
    
    opt=mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm=algo, max_iters=1000, bias=True,
                             is_classifier= True, learning_rate=0.9, early_stopping=True, clip_max= 5,
                             max_attempts=100, random_state= 100)
    opt.fit(x_train_norm, y_train)
    
    # XXX
    # TODO: Test its accuracy on the training set using the accuracy_score method.
    # TODO: Test its accuracy on the test set using the accuracy_score method.
    # XXX
    
    y_predict_test=opt.predict(x_test_norm)
    y_predict_train=opt.predict(x_train_norm)
    
    train=accuracy_score(y_train,y_predict_train.round())
    test=accuracy_score(y_test,y_predict_test.round())
    
    trainAccuracy.append(train*100)
    testAccuracy.append(test*100)
    #treeDepth.append(i);
        
    print("trainAccuracy:", trainAccuracy)
    print("testAccuracy:", testAccuracy)
    
    trainAccuracy_nSize=[]
    testAccuracy_nSize=[]
    NetworkSize=[]
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    for i in range(150, 250, 100):
        opt_nSize=mlrose.NeuralNetwork(hidden_nodes=[i], activation='relu', algorithm=algo, max_iters=1000, bias=True,
                             is_classifier= True, learning_rate=0.9, early_stopping=True, clip_max= 5,
                             max_attempts=100, random_state= 100)
        
        opt_nSize.fit(x_train, y_train)

        y_predict_test=opt_nSize.predict(x_test)
        y_predict_train=opt_nSize.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
                
        trainAccuracy_nSize.append(train*100)
        testAccuracy_nSize.append(test*100)
        NetworkSize.append(i);
    
    trainAccuracyTrainSizeTuned=[]
    testAccuracyTrainSizeTuned=[]
    cvAccuracyTrainSizeTuned=[]
    TrainSize=[]
    trainTime=[]
    trainMemory=[]
    
    for i in range(70, 75, 5):
        x_train, x_test, y_train_temp, y_test_temp = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        opt_tuned=mlrose.NeuralNetwork(hidden_nodes=[100], activation='relu', algorithm=algo, max_iters=1000, bias=True,
                             is_classifier= True, learning_rate=0.9, early_stopping=True, clip_max= 5,
                             max_attempts=100, random_state= 100)
        
        
        x_data_norm = normalize(x_data, norm='l2')
        x_train_norm, x_test_norm, y_train_temp, y_test_temp = train_test_split(x_data_norm, y_data, test_size=0.25, random_state=100)
        one_hot=OneHotEncoder()
        y_train=one_hot.fit_transform(y_train_temp.values.reshape(-1,1)).todense()
        y_test=one_hot.fit_transform(y_test_temp.values.reshape(-1,1)).todense()
        
        
        scores = cross_val_score(opt_tuned, x_test, y_test, cv=10)
        cvAccuracyTrainSizeTuned.append(scores.mean()*100)
        
        start_time = time.time()
        opt_tuned.fit(x_train, y_train)
        fit_time=time.time()-start_time
        trainTime.append(fit_time);
        y_predict_test=opt_tuned.predict(x_test)
        y_predict_train=opt_tuned.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
                
        trainAccuracyTrainSizeTuned.append(train*100)
        testAccuracyTrainSizeTuned.append(test*100)
        TrainSize.append(i);
        process = (psutil.Process(os.getpid()))
        trainMemory.append((process.memory_info().rss)/1e9)
        
        if i==75:
            print("Train Score", train)
            print("CV Score", scores.mean())
            print("Test Score", test)
            print("Fit Time", fit_time)
            print("Memory Suage (MB)", (process.memory_info().rss)/1e6)
      


    #print(testAccuracyTrainSizeTuned)
    plt.figure('NNM_01')
    plt.title('Accuracy vs Hidden Layer Size', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Hidden Layer Size', fontsize=14);
    #plt.xlim(0, 100)
    plt.ylim(40, 65)
    plt.plot(NetworkSize, trainAccuracy_nSize, 'go', label='Train Accuracy')
    plt.plot(NetworkSize, testAccuracy_nSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('NNM_02')
    plt.title('Accuracy (Tuned) vs Training Dataset Size', fontsize=16);
    plt.ylabel('Accuracy (Tuned) (%)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim(40, 65)
    plt.plot(TrainSize, trainAccuracyTrainSizeTuned, 'go', label='Train Accuracy')
    plt.plot(TrainSize, testAccuracyTrainSizeTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('NNM_03')
    plt.title('Training Time vs Training Dataset Size', fontsize=16);
    plt.ylabel('Time (s)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    #plt.ylim()
    plt.plot(TrainSize, trainTime, 'o', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('NNM_04')
    plt.title('Memory Usage vs Training Dataset Size', fontsize=16);
    plt.ylabel('Memory (Giga Bytes)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim()
    plt.plot(TrainSize, trainMemory, 's', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    
    print("End of Neural Network Analysis"); 

'''    
    plt.figure();
    plt.title('Accuracy vs Maximum Tree Depth', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Maximum Tree Depth', fontsize=14);
    plt.plot(treeDepth, trainAccuracy, 'go', label='Train Accuracy')
    plt.plot(treeDepth, testAccuracy, 'rs' , label='Test Accuracy')
    plt.legend(loc='upper left', fontsize=11)
    plt.show()
'''
    
      

###### End of Neural Network  Analysis ######

######################################### Reading and Splitting the Data ############################################


dataset=pd.read_csv('letter_recognition_data.csv')

# drop columns with missing values
#dataset1=dataset1.drop(columns='stalkRoot');

# Separate out the x_data and y_data.
x_data = dataset.loc[:, dataset.columns != "class"]

y_data = dataset.loc[:, "class"]

#oh = OneHotEncoder()
#oh.fit(y_data_temp)
#y_data=oh.transform(y_data_temp)



if len(sys.argv)<2:
    print("Default Algorithm: Random Hill Climb")
    nnMlpAnalysis (x_data, y_data, 'random_hill_climb')   
elif sys.argv[1]=='RHC':
    nnMlpAnalysis (x_data, y_data, 'random_hill_climb')   
elif sys.argv[1]=='GA':
    nnMlpAnalysis (x_data, y_data, 'genetic_alg')   
elif sys.argv[1]=='SA':
    nnMlpAnalysis (x_data, y_data, 'simulated_annealing')   
else:
    print("Invalid classifier name!\nValid options are: RHC (default), GA, SA.")