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

# ######################################## Decision Tree Classifier Analysis ############################################

def decisionTreeAnalysis (x_train, x_test, y_train, y_test):
    print("Starting  Decision Tree Classifier Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    trainAccuracy=[]
    testAccuracy=[]
    treeDepth=[]
    for i in range(1, 30):
        dtc=tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=i, random_state=100)
        dtc.fit(x_train, y_train)
        y_predict_test=dtc.predict(x_test)
        y_predict_train=dtc.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracy.append(train*100)
        testAccuracy.append(test*100)
        treeDepth.append(i);
        
    
    feature_importances=pd.DataFrame(dtc.feature_importances_, index=x_train.columns, 
                                  columns=['Importance']).sort_values('Importance',ascending=False)
    print(feature_importances);
    
    print("trainAccuracy:", trainAccuracy)
    print("testAccuracy:", testAccuracy)
    
    param_grid={
        'max_depth': [1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
        'random_state': [100]
    }
    dtc_tuned=tree.DecisionTreeClassifier()
    grid_search_dtc=GridSearchCV(estimator = dtc_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)
    grid_search_dtc.fit(x_train, y_train)

    print("Best Parameters:", grid_search_dtc.best_params_)
    print("Best Score:", grid_search_dtc.cv_results_)
    best_grid = grid_search_dtc.best_estimator_
    
    #best_random=grid_search_adb.best_estimator_
    y_predict_test_tuned=best_grid.predict(x_test)
    y_predict_train_tuned=best_grid.predict(x_train)
    
    accuracy_train_adb_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
    accuracy_test_adb_tuned=accuracy_score(y_test,y_predict_test_tuned.round())
    
    print("trainAccuracyTuned:", accuracy_train_adb_tuned)
    print("testAccuracyTuned:", accuracy_test_adb_tuned)
    print("Best Grid (Decision Tree Classifier):", best_grid)
    
    plt.figure();
    plt.title('Accuracy vs Maximum Tree Depth', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Maximum Tree Depth', fontsize=14);
    plt.plot(treeDepth, trainAccuracy, 'go', label='Train Accuracy')
    plt.plot(treeDepth, testAccuracy, 'rs' , label='Test Accuracy')
    plt.legend(loc='upper left', fontsize=11)
    plt.show()
    
    print("End of  Decision Tree Classifier Analysis");   

###### End of Decision Tree Classifier Analysis ######
    

# ######################################## Ada Boost Classifier Analysis ############################################

def adaBoostAnalysis (x_train, x_test, y_train, y_test):
    print("Starting  Ada Boost Classifier Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    trainAccuracy=[]
    testAccuracy=[]
    treeDepth=[]
    
    adb=AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), random_state=100)        
    adb.fit(x_train, y_train)
    y_predict_test=adb.predict(x_test)
    y_predict_train=adb.predict(x_train)   
    train=accuracy_score(y_train,y_predict_train.round())
    test=accuracy_score(y_test,y_predict_test.round())
        
    trainAccuracy.append(train*100)
    testAccuracy.append(test*100)
    #treeDepth.append(i);
        
    print("trainAccuracy:", trainAccuracy)
    print("testAccuracy:", testAccuracy)
    
    feature_importances=pd.DataFrame(adb.feature_importances_, index=x_train.columns, 
                                  columns=['Importance']).sort_values('Importance',ascending=False)
    print(feature_importances)
    
    ###### Parameters optimization #####
    param_grid={
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.75, 0.9, 1.0],
        'random_state': [100],
        'base_estimator': [tree.DecisionTreeClassifier(max_depth=10),
                           tree.DecisionTreeClassifier(max_depth=15), 
                           tree.DecisionTreeClassifier(max_depth=20),
                           tree.DecisionTreeClassifier(max_depth=25)]
    }
    adb_tuned=AdaBoostClassifier()
    grid_search_adb=GridSearchCV(estimator = adb_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)
    grid_search_adb.fit(x_train, y_train)

    print("Best Parameters:", grid_search_adb.best_params_)
    print("Best Score:", grid_search_adb.cv_results_)
    best_grid = grid_search_adb.best_estimator_
    
    #best_random=grid_search_adb.best_estimator_
    y_predict_test_tuned=best_grid.predict(x_test)
    y_predict_train_tuned=best_grid.predict(x_train)
    
    accuracy_train_adb_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
    accuracy_test_adb_tuned=accuracy_score(y_test,y_predict_test_tuned.round())
    
    print("trainAccuracyTuned:", accuracy_train_adb_tuned)
    print("testAccuracyTuned:", accuracy_test_adb_tuned)
    print("Best Grid (Ada Boost Classifier):", best_grid)
    
    print("End of  Ada Boost Classifier Analysis"); 

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
    
      

###### End of Ada Boost Classifier Analysis ######

# ######################################## K-Nearest Neighbour Classifier Analysis ############################################

def kNearestNeighbourAnalysis (x_train, x_test, y_train, y_test):
    print("Starting  KNN Classifier Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    trainAccuracy=[]
    testAccuracy=[]
    treeDepth=[]
    
    knn=KNeighborsClassifier(algorithm='auto')        
    knn.fit(x_train, y_train)
    y_predict_test=knn.predict(x_test)
    y_predict_train=knn.predict(x_train)   
    train=accuracy_score(y_train,y_predict_train.round())
    test=accuracy_score(y_test,y_predict_test.round())
        
    trainAccuracy.append(train*100)
    testAccuracy.append(test*100)
    #treeDepth.append(i);
        
    print("trainAccuracy:", trainAccuracy)
    print("testAccuracy:", testAccuracy)
    
    #feature_importances=pd.DataFrame(knn.feature_importances_, index=x_train.columns, 
                                 # columns=['Importance']).sort_values('Importance',ascending=False)
    #print(feature_importances)
    
    ###### Parameters optimization #####
    param_grid={
        'n_neighbors': [1, 3, 5, 10, 15, 20, 25],
        'p': [1, 2],
        'algorithm': ['auto']
    }
    knn_tuned=KNeighborsClassifier()
    grid_search_knn=GridSearchCV(estimator = knn_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)
    grid_search_knn.fit(x_train, y_train)
    
    best_grid = grid_search_knn.best_estimator_


    print("Best Parameters:", grid_search_knn.best_params_)
    print("Best Score:", grid_search_knn.cv_results_)
    
    #np.savetxt("knn_cv.csv", grid_search_knn.cv_results_, delimiter=",")
    #np.savetxt("knn_best_params.csv", grid_search_knn.best_params_, delimiter=",")
    #np.savetxt("knn_best_grid.csv", grid_search_knn.best_estimator_, delimiter=",")
        
    #best_random=grid_search_adb.best_estimator_
    y_predict_test_tuned=best_grid.predict(x_test)
    y_predict_train_tuned=best_grid.predict(x_train)
    
    accuracy_train_knn_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
    accuracy_test_knn_tuned=accuracy_score(y_test,y_predict_test_tuned.round())
    
    print("trainAccuracyTuned:", accuracy_train_knn_tuned)
    print("testAccuracyTuned:", accuracy_test_knn_tuned)
    print("Best Grid (K-Nearest Neighbour Classifier):", best_grid)
    
    print("End of  KNN Classifier Analysis"); 

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
    
      

###### End of KNN Classifier Analysis ######


# ######################################## Support Vector Machines Classifier Analysis ############################################

def svmAnalysis (x_train, x_test, y_train, y_test):
    print("Starting  SVM (C-Support Vector Classification) Analysis.....");
    
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
    
    svm_c=SVC(gamma='auto', random_state=100)
    svm_c.fit(x_train_norm, y_train)
    
    # XXX
    # TODO: Test its accuracy on the training set using the accuracy_score method.
    # TODO: Test its accuracy on the test set using the accuracy_score method.
    # XXX
    
    y_predict_test=svm_c.predict(x_test_norm)
    y_predict_train=svm_c.predict(x_train_norm)
    
    train=accuracy_score(y_train,y_predict_train.round())
    test=accuracy_score(y_test,y_predict_test.round())
    
    trainAccuracy.append(train*100)
    testAccuracy.append(test*100)
    #treeDepth.append(i);
        
    print("trainAccuracy:", trainAccuracy)
    print("testAccuracy:", testAccuracy)
    
    
    param_grid={
            'C': [0.001, 0.01, 0.1, 1, 10],
            'degree': [1 ,3, 5, 10],
            'random_state': [100],
            'gamma': ['scale'],
            'kernel': ['rbf', 'linear']
    }
    
    svc_tuned=SVC()
    grid_search_svc=GridSearchCV(estimator = svc_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)
    grid_search_svc.fit(x_train_norm, y_train)
    
    best_grid = grid_search_svc.best_estimator_


    print("Best Parameters:", grid_search_svc.best_params_)
    print("Best Score:", grid_search_svc.cv_results_)
    
        
    #best_random=grid_search_adb.best_estimator_
    y_predict_test_tuned=best_grid.predict(x_test_norm)
    y_predict_train_tuned=best_grid.predict(x_train_norm)
    
    accuracy_train_svc_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
    accuracy_test_svc_tuned=accuracy_score(y_test,y_predict_test_tuned.round())
    
    print("trainAccuracyTuned:", accuracy_train_svc_tuned)
    print("testAccuracyTuned:", accuracy_test_svc_tuned)
    print("Best Grid (SVC Classifier):", best_grid)
    
    
    print("End of  SVM (C-SVC) Classifier Analysis"); 

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
    
      

###### End of SVM Classifier Analysis ######


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
            'hidden_layer_sizes': [(25,), (50,), (100,), (150,)],
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

dataset=pd.read_csv('letter_recognition_data.csv')

# drop columns with missing values
#dataset1=dataset1.drop(columns='stalkRoot');

# Separate out the x_data and y_data.
x_data = dataset.loc[:, dataset.columns != "class"]

y_data_temp = dataset.loc[:, "class"]

le = LabelEncoder()
le.fit(y_data_temp)
y_data=le.transform(y_data_temp)

# TODO: Split 75% of the data into training and 25% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)

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
###### Start of  Random Forest Classifier Analysis ######

def random_forest_analysis (x_train, x_test, y_train, y_test):
    rfc=RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_predict_test=rfc.predict(x_test)
    y_predict_train=rfc.predict(x_train)

    accuracy_train_rfc=accuracy_score(y_train,y_predict_train.round())
    accuracy_test_rfc=accuracy_score(y_test,y_predict_test.round())

    feature_importances=pd.DataFrame(rfc.feature_importances_, index=x_train.columns, 
                                  columns=['Importance']).sort_values('Importance',ascending=False)
    print(feature_importances);

    param_grid={
            'n_estimators': [10, 100, 1000],
            'max_depth': [1, 10, 100]
        }
    rfc_tuned=RandomForestClassifier()
    grid_search_rfc=GridSearchCV(estimator = rfc_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)
    grid_search_rfc.fit(x_train, y_train)

    print("Best Parameters:", grid_search_rfc.best_params_)
    print("Best Score:", grid_search_rfc.cv_results_)
    best_grid = grid_search.best_estimator_
    
    best_random=grid_search_rfc.best_estimator_
    y_predict_test_tuned=best_random.predict(x_test)
    y_predict_train_tuned=best_random.predict(x_train)
    
    accuracy_train_rfc_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
    accuracy_test_rfc_tuned=accuracy_score(y_test,y_predict_test_tuned.round())
    

random_forest_analysis (x_train, x_test, y_train, y_test)
    

'''
############### End of Random Forest Classifier Analysis ##################

"""
# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX

lm=LinearRegression()
lm.fit(x_train, y_train)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX

y_predict_test=lm.predict(x_test)
y_predict_train=lm.predict(x_train)

#print(y_predict.shape)

accuracy_train_lm=accuracy_score(y_train,y_predict_train.round())
accuracy_test_lm=accuracy_score(y_test,y_predict_test.round())

print(accuracy_train_lm)
print(accuracy_test_lm)

# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

#rfc=RandomForestClassifier(n_estimators=100)
rfc=RandomForestClassifier()
rfc.fit(x_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_predict_test=rfc.predict(x_test)
y_predict_train=rfc.predict(x_train)

accuracy_train_rfc=accuracy_score(y_train,y_predict_train.round())
accuracy_test_rfc=accuracy_score(y_test,y_predict_test.round())

print(accuracy_train_rfc)
print(accuracy_test_rfc)

# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX

feature_importances=pd.DataFrame(rfc.feature_importances_, index=x_train.columns, 
                                  columns=['Importance']).sort_values('Importance',ascending=False)
print(feature_importances)
print("Most important feature: "+feature_importances.index[0]+
      "\nLeast important feature: "+feature_importances.index[len(feature_importances)-1])
# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

param_grid={
            'n_estimators': [10, 100, 1000],
            'max_depth': [1, 10, 100]
        }

rfc_tuned=RandomForestClassifier()
grid_search_rfc=GridSearchCV(estimator = rfc_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)

grid_search_rfc.fit(x_train, y_train)

print("Best Parameters:", grid_search_rfc.best_params_)
print("Best Score:", grid_search_rfc.cv_results_)
#best_grid = grid_search.best_estimator_

best_random=grid_search_rfc.best_estimator_
y_predict_test_tuned=best_random.predict(x_test)
y_predict_train_tuned=best_random.predict(x_train)

accuracy_train_rfc_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
accuracy_test_rfc_tuned=accuracy_score(y_test,y_predict_test_tuned.round())

print(accuracy_train_rfc_tuned)
print(accuracy_test_rfc_tuned)


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX
x_data_norm = normalize(x_data, norm='l2')
x_train_norm, x_test_norm, y_train, y_test = train_test_split(x_data_norm, y_data, test_size=0.30, random_state=100)

svm_c=SVC()
svm_c.fit(x_train_norm, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_predict_test=svm_c.predict(x_test_norm)
y_predict_train=svm_c.predict(x_train_norm)

accuracy_train_svc=accuracy_score(y_train,y_predict_train.round())
accuracy_test_svc=accuracy_score(y_test,y_predict_test.round())

print(accuracy_train_svc)
print(accuracy_test_svc)


# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

param_grid={
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear']
        }

svc_tuned=SVC()
grid_search_svc=GridSearchCV(estimator = svc_tuned, param_grid = param_grid, cv = 10, n_jobs = -1)

grid_search_svc.fit(x_train_norm, y_train)

print("Best Parameters:", grid_search_svc.best_params_)
print("Best Score:", grid_search_svc.best_score_)
print("Results:", grid_search_svc.cv_results_)


#best_grid = grid_search.best_estimator_

best_random=grid_search_svc.best_estimator_
y_predict_test_tuned=best_random.predict(x_test_norm)
y_predict_train_tuned=best_random.predict(x_train_norm)

accuracy_train_svc_tuned=accuracy_score(y_train,y_predict_train_tuned.round())
accuracy_test_svc_tuned=accuracy_score(y_test,y_predict_test_tuned.round())

print(accuracy_train_svc_tuned)
print(accuracy_test_svc_tuned)

# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX

pca=PCA(n_components=10, svd_solver='full')
pca.fit(x_data)
print(pca.explained_variance_)
print(pca.singular_values_)
"""