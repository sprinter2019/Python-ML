## MAchine Learning - Assignment 1
## Georgia Institute of Technology
## Applying ML supervised algorithms to interesting datasets

import matplotlib.pyplot as plt
import sys

import numpy as np
import pandas as pd
import time

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
#from sklearn.decomposition import PCA



# ######################################## Neural Network (MLP Classifier) Analysis ############################################

def nnMlpAnalysis (x_train, x_test, y_train, y_test):
    print("Starting  Neural Network (Multi-layer Perceptron Classifier) Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    trainAccuracy=[]
    testAccuracy=[]
    treeDepth=[]
    
    # XXX
    # TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
    # TODO: Create a SVC classifier and train it.
    # XXX
    x_data_norm = normalize(x_data, norm='l2')
    x_train_norm, x_test_norm, y_train, y_test = train_test_split(x_data_norm, y_data, test_size=0.25, random_state=100)
    
    mlp=MLPClassifier(learning_rate='adaptive', early_stopping=True, tol=1e-5, n_iter_no_change=25, random_state=100)
    mlp.fit(x_train_norm, y_train)
    
    # XXX
    # TODO: Test its accuracy on the training set using the accuracy_score method.
    # TODO: Test its accuracy on the test set using the accuracy_score method.
    # XXX
    
    y_predict_test=mlp.predict(x_test_norm)
    y_predict_train=mlp.predict(x_train_norm)
    
    train=accuracy_score(y_train,y_predict_train.round())
    test=accuracy_score(y_test,y_predict_test.round())
    
    trainAccuracy.append(train*100)
    testAccuracy.append(test*100)
    #treeDepth.append(i);
        
    print("trainAccuracy:", trainAccuracy)
    print("testAccuracy:", testAccuracy)
    
    
    param_grid={
            'alpha': [1e-7, 1e-5],
            'early_stopping': [True],
            'tol': [1e-5],
            'n_iter_no_change': [25],
            'hidden_layer_sizes': [(25,), (50,), (100,), (150,), (300, )],
            'learning_rate':['adaptive'],
            'random_state': [100],
            'activation': ['relu', 'tanh', 'identity'],
            'solver': ['adam', 'lbfgs', 'sgd']
    }
    
    mlp_tuned=MLPClassifier()
    grid_search_mlp=GridSearchCV(estimator = mlp_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)
    grid_search_mlp.fit(x_train_norm, y_train)
    
    best_grid = grid_search_mlp.best_estimator_


    print("Best Parameters:", grid_search_mlp.best_params_)
    print("Best Score:", grid_search_mlp.cv_results_)
    
        
    #best_random=grid_search_adb.best_estimator_
    y_predict_test_tuned=best_grid.predict(x_test_norm)
    y_predict_train_tuned=best_grid.predict(x_train_norm)
    
    accuracy_train_mlp_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
    accuracy_test_mlp_tuned=accuracy_score(y_test,y_predict_test_tuned.round())
    
    print("trainAccuracyTuned:", accuracy_train_mlp_tuned)
    print("testAccuracyTuned:", accuracy_test_mlp_tuned)
    print("Best Grid (MLP (NN) Classifier):", best_grid)
    
    
    print("End of  Neural Network (Multi-layer Perceptron Classifier) Analysis"); 

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

dataset=pd.read_csv('EEG_Eye_State.csv')

# Separate out the x_data and y_data.
x_data = dataset.loc[:, dataset.columns != "class"]
y_data = dataset.loc[:, "class"]

# TODO: Split 75% of the data into training and 25% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)

nnMlpAnalysis (x_train, x_test, y_train, y_test)

'''
if len(sys.argv)<2:
    print("Running default classifier: Decision Tree Classifier")
    decisionTreeAnalysis (x_train, x_test, y_train, y_test)   
elif sys.argv[1]=='DTC':
    decisionTreeAnalysis (x_train, x_test, y_train, y_test)
elif sys.argv[1]=='ADB':
    adaBoostAnalysis (x_train, x_test, y_train, y_test)
elif sys.argv[1]=='KNN':
    kNearestNeighbourAnalysis (x_train, x_test, y_train, y_test)
elif sys.argv[1]=='SVM':
    svmAnalysis (x_train, x_test, y_train, y_test)
elif sys.argv[1]=='NNM':
    nnMlpAnalysis (x_train, x_test, y_train, y_test)
else:
    print("Invalid classifier name!\nValid options are: DTC (default), ADB, SVM, KNN, NNC.")
###### End of  Decision Tree Classifier Analysis ######


'''