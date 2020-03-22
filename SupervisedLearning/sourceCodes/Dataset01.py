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

from sklearn.model_selection import validation_curve
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

def decisionTreeAnalysis (x_data, y_data):
    #pool= Pool(processes=(cpu_count() - 1))
    print("Starting  Decision Tree Classifier Analysis.....");
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    trainAccuracy=[]
    testAccuracy=[]
    treeDepth=[]
    for i in range(1, 50):
        dtc=tree.DecisionTreeClassifier(criterion="gini", max_depth=i, random_state=100)
        dtc.fit(x_train, y_train)
        y_predict_test=dtc.predict(x_test)
        y_predict_train=dtc.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracy.append(train*100)
        testAccuracy.append(test*100)
        treeDepth.append(i);
        
    trainAccuracyTrainSize=[]
    testAccuracyTrainSize=[]
    cvAccuracyTrainSizeTuned=[]    
    TrainSize=[]
    
    trainTime=[]
    trainMemory=[]
    
    for i in range(5, 95, 5):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        dtc=tree.DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=100)
       
        scores = cross_val_score(dtc, x_test, y_test, cv=10)
        cvAccuracyTrainSizeTuned.append(scores.mean()*100)
        start_time = time.time()
        dtc.fit(x_train, y_train)
        fit_time=time.time()-start_time
        trainTime.append(fit_time);
        y_predict_test=dtc.predict(x_test)
        y_predict_train=dtc.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracyTrainSize.append(train*100)
        testAccuracyTrainSize.append(test*100)
        process = (psutil.Process(os.getpid()))
        TrainSize.append(i);
        trainMemory.append((process.memory_info().rss)/1e9)
        if i==75:
            print("Train Score", train)
            print("CV Score", scores.mean())
            print("Test Score", test)
            print("Fit Time", fit_time)
            print("Memory Suage (MB)", (process.memory_info().rss)/1e6)
    
    #start_time = time.time()
    #feature_importances=pd.DataFrame(dtc.feature_importances_, index=x_train.columns, 
                                  #columns=['Importance']).sort_values('Importance',ascending=False)
    #print(feature_importances);
    
    #print("trainAccuracy:", trainAccuracy)
    #print("testAccuracy:", testAccuracy)
    
    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss)
    
    plt.figure('DTC 01')
    plt.title('Accuracy vs Maximum Tree Depth', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Maximum Tree Depth', fontsize=14);
    plt.ylim(50, 110)
    plt.plot(treeDepth, trainAccuracy, 'go', label='Train Accuracy')
    plt.plot(treeDepth, testAccuracy, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('DTC 02')
    plt.title('Accuracy vs Training Dataset Size', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim(60, 110)
    plt.plot(TrainSize, trainAccuracyTrainSize, 'go', label='Train Accuracy')
    plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('DTC 03')
    plt.title('Training Time vs Training Dataset Size', fontsize=16);
    plt.ylabel('Time (s)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    #plt.ylim(0)
    plt.plot(TrainSize, trainTime, 'o', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('DTC 04')
    plt.title('Memory Usage vs Training Dataset Size', fontsize=16);
    plt.ylabel('Memory (Giga Bytes)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim()
    plt.plot(TrainSize, trainMemory, 's', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    
    
    print("End of  Decision Tree Classifier Analysis");

###### End of Decision Tree Classifier Analysis ######
    

# ######################################## Ada Boost Classifier Analysis ############################################

def adaBoostAnalysis (x_data, y_data):
    print("Starting  Ada Boost Classifier Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    trainAccuracy=[]
    testAccuracy=[]
    #treeDepth=[]
    
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
    
    
    trainAccuracyEstimatorSizeTuned=[]
    testAccuracyEstimatorSizeTuned=[]
    EstimatorSize=[]
    for i in range(50, 50, 50):
        adb_Estimator=AdaBoostClassifier(algorithm='SAMME.R', 
                               base_estimator=tree.DecisionTreeClassifier(class_weight=None, 
                                                criterion='gini', max_depth=10,
                                                max_features=None,
                                                max_leaf_nodes=None,
                                                min_impurity_decrease=0.0,
                                                min_impurity_split=None,
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                min_weight_fraction_leaf=0.0,
                                                presort=False,
                                                random_state=100,
                                                splitter='best'),
                                                learning_rate=0.9, n_estimators=i, random_state=100)
        
        adb_Estimator.fit(x_train, y_train)
        y_predict_test=adb_Estimator.predict(x_test)
        y_predict_train=adb_Estimator.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracyEstimatorSizeTuned.append(train*100)
        testAccuracyEstimatorSizeTuned.append(test*100)
        EstimatorSize.append(i);
    
    #feature_importances=pd.DataFrame(adb.feature_importances_, index=x_train.columns, 
                                  #columns=['Importance']).sort_values('Importance',ascending=False)
    #print(feature_importances)
    
    trainAccuracyTrainSizeTuned=[]
    testAccuracyTrainSizeTuned=[]
    cvAccuracyTrainSizeTuned=[]
    TrainSize=[]
    trainTime=[]
    trainMemory=[]
    for i in range(5, 95, 5):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        adb_tuned=AdaBoostClassifier(algorithm='SAMME.R', 
                               base_estimator=tree.DecisionTreeClassifier(class_weight=None, 
                                                criterion='gini', max_depth=10,
                                                max_features=None,
                                                max_leaf_nodes=None,
                                                min_impurity_decrease=0.0,
                                                min_impurity_split=None,
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                min_weight_fraction_leaf=0.0,
                                                presort=False,
                                                random_state=100,
                                                splitter='best'),
                                                learning_rate=0.9, n_estimators=100, random_state=100)
        
        scores = cross_val_score(adb_tuned, x_test, y_test, cv=10)
        cvAccuracyTrainSizeTuned.append(scores.mean()*100)
        start_time = time.time()
        adb_tuned.fit(x_train, y_train)
        fit_time=time.time()-start_time
        trainTime.append(fit_time);
        y_predict_test=adb_tuned.predict(x_test)
        y_predict_train=adb_tuned.predict(x_train)   
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
        #print(trainTime)


    plt.figure('ADB_01')
    plt.title('Accuracy vs Estimator Size', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Estimator(n_estimator) Size', fontsize=14);
    #plt.xlim(0, 350)
    plt.ylim(90, 105)
    plt.plot(EstimatorSize, trainAccuracyEstimatorSizeTuned, 'go', label='Train Accuracy')
    plt.plot(EstimatorSize, testAccuracyEstimatorSizeTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)

    plt.figure('ADB_02')
    plt.title('Accuracy (Tuned) vs Training Dataset Size', fontsize=16);
    plt.ylabel('Accuracy (Tuned) (%)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim(60, 110)
    plt.plot(TrainSize, trainAccuracyTrainSizeTuned, 'go', label='Train Accuracy')
    plt.plot(TrainSize, testAccuracyTrainSizeTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('ADB_03')
    plt.title('Training Time vs Training Dataset Size', fontsize=16);
    plt.ylabel('Time (s)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim()
    plt.plot(TrainSize, trainTime, 'o', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('ADB_04')
    plt.title('Memory Usage vs Training Dataset Size', fontsize=16);
    plt.ylabel('Memory (Giga Bytes)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim()
    plt.plot(TrainSize, trainMemory, 's', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    
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

def kNearestNeighbourAnalysis (x_data, y_data):
    print("Starting  KNN Classifier Analysis.....");
    process1 = (psutil.Process(os.getpid()))
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    trainAccuracy=[]
    testAccuracy=[]
    #treeDepth=[]
    
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
    
    trainAccuracyKSizeTuned=[]
    testAccuracyKSizeTuned=[]
    KSize=[]

    for i in range(1, 20):
        knn_k=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=i, p=2,
                     weights='uniform')
        
        knn_k.fit(x_train, y_train)
        y_predict_test=knn_k.predict(x_test)
        y_predict_train=knn_k.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        
        trainAccuracyKSizeTuned.append(train*100)
        testAccuracyKSizeTuned.append(test*100)
        KSize.append(i);
    
    #feature_importances=pd.DataFrame(knn.feature_importances_, index=x_train.columns, 
                                 # columns=['Importance']).sort_values('Importance',ascending=False)
    #print(feature_importances)
    
    ###### Parameters optimization #####
    trainAccuracyTrainSizeTuned=[]
    testAccuracyTrainSizeTuned=[]
    cvAccuracyTrainSizeTuned=[]
    TrainSize=[]
    trainTime=[]
    trainMemory=[]
    for i in range(5, 95, 5):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        knn_tuned=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')
        
        scores = cross_val_score(knn_tuned, x_test, y_test, cv=10)
        cvAccuracyTrainSizeTuned.append(scores.mean()*100)
        start_time = time.time()
        knn_tuned.fit(x_train, y_train)
        fit_time=time.time()-start_time
        trainTime.append(fit_time);
        y_predict_test=knn_tuned.predict(x_test)
        y_predict_train=knn_tuned.predict(x_train)   
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

    #process2 = (psutil.Process(os.getpid()))
    
    #print("Memory Usage: ", (((process2.memory_info().rss)-(process1.memory_info().rss))/1e9))
    
    plt.figure('KNN_01')
    plt.title('Accuracy vs Number of Neighbours', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Number of neighbours', fontsize=14);
    plt.xlim(0, 21)
    plt.ylim(85,101)
    plt.plot(KSize, trainAccuracyKSizeTuned, 'go', label='Train Accuracy')
    plt.plot(KSize, testAccuracyKSizeTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)

    plt.figure('KNN_02')
    plt.title('Accuracy (Tuned) vs Training Dataset Size', fontsize=16);
    plt.ylabel('Accuracy (Tuned) (%)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    #plt.ylim(90,101)
    plt.plot(TrainSize, trainAccuracyTrainSizeTuned, 'go', label='Train Accuracy')
    plt.plot(TrainSize, testAccuracyTrainSizeTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('KNN_03')
    plt.title('Training Time vs Training Dataset Size', fontsize=16);
    plt.ylabel('Time (s)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    #plt.ylim()
    plt.plot(TrainSize, trainTime, 'o', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('KNN_04')
    plt.title('Memory Usage vs Training Dataset Size', fontsize=16);
    plt.ylabel('Memory (Giga Bytes)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    #plt.ylim()
    plt.plot(TrainSize, trainMemory, 's', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    
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

def svmAnalysis (x_data, y_data):
    print("Starting  SVM (C-Support Vector Classification) Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    trainAccuracy=[]
    testAccuracy=[]
    #treeDepth=[]
    
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
    
    trainAccuracyCTuned=[]
    testAccuracyCTuned=[]
    CSize=[]
    trainTime=[]
    trainMemory=[]
    for i in range(1, 151, 10):
        #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        svm_Ctuned=SVC(C=i, cache_size=200, class_weight=None, coef0=0.0, 
                      decision_function_shape='ovr', degree=3, gamma='scale', 
                      kernel='rbf', max_iter=-1, probability=False, random_state=100, 
                      shrinking=True, tol=0.001, verbose=False)
        
        svm_Ctuned.fit(x_train, y_train)
        y_predict_test=svm_Ctuned.predict(x_test)
        y_predict_train=svm_Ctuned.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        trainAccuracyCTuned.append(train*100)
        testAccuracyCTuned.append(test*100)
        CSize.append(i);
    
    trainAccuracyTrainSizeTuned=[]
    testAccuracyTrainSizeTuned=[]
    cvAccuracyTrainSizeTuned=[]
    TrainSize=[]
    trainTime=[]
    trainMemory=[]
    for i in range(5, 95, 5):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        svm_tuned=SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, 
                      decision_function_shape='ovr', degree=3, gamma='scale', 
                      kernel='rbf', max_iter=-1, probability=False, random_state=100, 
                      shrinking=True, tol=0.001, verbose=False)
        
        scores = cross_val_score(svm_tuned, x_test, y_test, cv=10)
        cvAccuracyTrainSizeTuned.append(scores.mean()*100)
        
        start_time = time.time()
        svm_tuned.fit(x_train, y_train)
        fit_time=time.time()-start_time
        trainTime.append(fit_time);
        
        y_predict_test=svm_tuned.predict(x_test)
        y_predict_train=svm_tuned.predict(x_train)   
        train=accuracy_score(y_train,y_predict_train.round())
        test=accuracy_score(y_test,y_predict_test.round())
        trainAccuracyTrainSizeTuned.append(train*100)
        testAccuracyTrainSizeTuned.append(test*100)
        TrainSize.append(i);
        process = (psutil.Process(os.getpid()))
        trainMemory.append((process.memory_info().rss)/1e6)
        if i==75:
            print("Train Score", train)
            print("CV Score", scores.mean())
            print("Test Score", test)
            print("Fit Time", fit_time)
            print("Memory Suage (MB)", (process.memory_info().rss)/1e6)

    plt.figure('SVM_01')
    plt.title('Accuracy vs Penalty Parameter (C)', fontsize=16);
    plt.ylabel('Accuracy (%)', fontsize=14);
    plt.xlabel('Penalty Parameter (C)', fontsize=14);
    plt.xlim(0, 160)
    plt.ylim(50, 70)
    plt.plot(CSize, trainAccuracyCTuned, 'go', label='Train Accuracy')
    plt.plot(CSize, testAccuracyCTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('SVM_02')
    plt.title('Accuracy (Tuned) vs Training Dataset Size', fontsize=16);
    plt.ylabel('Accuracy (Tuned) (%)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim(40, 80)
    plt.plot(TrainSize, trainAccuracyTrainSizeTuned, 'go', label='Train Accuracy')
    plt.plot(TrainSize, testAccuracyTrainSizeTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('SVM_03')
    plt.title('Training Time vs Training Dataset Size', fontsize=16);
    plt.ylabel('Time (s)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim()
    plt.plot(TrainSize, trainTime, 'o', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show(block=False)
    
    plt.figure('SVM_04')
    plt.title('Memory Usage vs Training Dataset Size', fontsize=16);
    plt.ylabel('Memory (Giga Bytes)', fontsize=14);
    plt.xlabel('Training Size (%)', fontsize=14);
    plt.xlim(0, 100)
    plt.ylim()
    plt.plot(TrainSize, trainMemory, 's', label='Training Time')
    #plt.plot(TrainSize, testAccuracyTrainSize, 'rs' , label='Test Accuracy')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    
    
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

def nnMlpAnalysis (x_data, y_data):
    print("Starting  Neural Network (Multi-layer Perceptron Classifier) Analysis.....");
    
    ### Pruning: Detremine max_depth that best optimizes the algorithm
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
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
    
    trainAccuracy_nSize=[]
    testAccuracy_nSize=[]
    NetworkSize=[]
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    for i in range(50, 500, 50):
        mlp_nSize=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(i,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
        
        mlp_nSize.fit(x_train, y_train)

        y_predict_test=mlp_nSize.predict(x_test)
        y_predict_train=mlp_nSize.predict(x_train)   
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
    
    for i in range(5, 95, 5):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(100-i)/100, random_state=100)
        mlp_tuned=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
        
        scores = cross_val_score(mlp_tuned, x_test, y_test, cv=10)
        cvAccuracyTrainSizeTuned.append(scores.mean()*100)
        
        start_time = time.time()
        mlp_tuned.fit(x_train, y_train)
        fit_time=time.time()-start_time
        trainTime.append(fit_time);
        y_predict_test=mlp_tuned.predict(x_test)
        y_predict_train=mlp_tuned.predict(x_train)   
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

if len(sys.argv)<2:
    print("Running default classifier: Decision Tree Classifier")
    decisionTreeAnalysis (x_data, y_data)   
elif sys.argv[1]=='DTC':
    decisionTreeAnalysis (x_data, y_data)
elif sys.argv[1]=='ADB':
    adaBoostAnalysis (x_data, y_data)
elif sys.argv[1]=='KNN':
    kNearestNeighbourAnalysis (x_data, y_data)
elif sys.argv[1]=='SVM':
    svmAnalysis (x_data, y_data)
elif sys.argv[1]=='NNM':
    nnMlpAnalysis (x_data, y_data)
else:
    print("Invalid classifier name!\nValid options are: DTC (default), ADB, SVM, KNN, NNC.")
