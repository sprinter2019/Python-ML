## MAchine Learning - Assignment 3
## Georgia Institute of Technology
## Applying ML unsupervised algorithms to interesting datasets

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
#from sklearn.metrics import accuracy_score, classification_report
import sklearn.metrics
#from sklearn import tree
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, adjusted_mutual_info_score
from  sklearn import mixture
#from sklearn import ensemble
#from sklearn.svm import SVC
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report



## Ref: https://towardsdatascience.com/separating-mixed-signals-with-independent-component-analysis-38205188f2f4
def kurtosis(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3
    #return kurt, skew, var, mean
    return kurt

# ######################################## Support KMeans Clustering Analysis ############################################

def clusteringAnalysisKMeans (x_data, y_data) :
    print("Starting  KMeans Clustering Analysis.....");
    
    kClusters=[]
    ssError=[]
    hmScoreKM=[]
    cmScoreKM=[]
    amiScoreKM=[]
    for k in range(1, 50):
        
        kmeans = KMeans(n_clusters=k, random_state=0).fit(x_data)
        
        #Y_labels[]= kmeans.labels_
        homogeneityScore=homogeneity_score(y_data, kmeans.labels_)
        hmScoreKM.append(homogeneityScore)
        
        completenessScore=completeness_score(y_data, kmeans.labels_)
        cmScoreKM.append(completenessScore)
        
        adjustedMIScore=adjusted_mutual_info_score(y_data, kmeans.labels_)
        amiScoreKM.append(adjustedMIScore)
        
        #sse[k] = kmeans.inertia_
        ssError.append(kmeans.inertia_)
        kClusters.append(k)
        

    dir = os.path.join("./","Dataset02_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    plt.figure('KM_01')
    plt.title('SSE vs Number of Clusters', fontsize=16);
    plt.ylabel('SSE', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, ssError, 'go--', linewidth=3, markersize=8, label='SS Error')
   # plt.plot(kClusters, testAccuracyCTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/KMeansSSE.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure('KM_02')
    plt.title('Scores vs Number of Clusters', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(kClusters, cmScoreKM, 'b*--', linewidth=3 , markersize=8, label='Completeness Score')
    plt.plot(kClusters, amiScoreKM, 'r--', linewidth=3 , markersize=8, label='AMI Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/KMeansScores.png', bbox_inches='tight')
    plt.show()
    

    print("End of  KMeans Clustering Analysis"); 


def clusteringAnalysisEMaximization (x_data, y_data):
    print("Starting  Empectation Maximization Analysis.....");
    
    mComponents=[]    

    scoreSamplesEM=[]    
    aicSCoreEM=[]
    bicSCoreEM=[]
    for k in range(1, 50):
        
        ##covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
        gmmEM = mixture.GaussianMixture(n_components=k, covariance_type='full').fit(x_data)
        
        #Y_labels[]= kmeans.labels_
#        homogeneityScore=homogeneity_score(y_data, gmmEM.labels_)
#        hmScoreKM.append(homogeneityScore)
        
#        completenessScore=completeness_score(y_data, gmmEM.labels_)
#        cmScoreKM.append(completenessScore)
        
#        adjustedMIScore=adjusted_mutual_info_score(y_data, gmmEM.labels_)
#        amiScoreKM.append(adjustedMIScore)
        
        #sse[k] = kmeans.inertia_
        
        scoreSamples=gmmEM.score(x_data)
        scoreSamplesEM.append(scoreSamples)
        
        aicScore=gmmEM.aic(x_data)
        aicSCoreEM.append(aicScore)
        
        bicScore=gmmEM.bic(x_data)
        bicSCoreEM.append(bicScore)
        
        
#        ssError.append(gmmEM.inertia_)
        mComponents.append(k)
        
    dir = os.path.join("./","Dataset02_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)

    
    plt.figure('EM_01')
    plt.title('Average Log Likelihood vs Number of Components', fontsize=16);
    plt.ylabel('Avg. Log Likelihood', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, scoreSamplesEM, 'go-', linewidth=3 , markersize=8, label='Avg. Log Likelihood')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/EMProbability.png', bbox_inches='tight')
    plt.show(block=False)
    
    plt.figure('EM_02')
    plt.title('Scores vs Number of Components', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, aicSCoreEM, 'bs-', linewidth=3 , markersize=8, label='AIC Score')
    plt.plot(mComponents, bicSCoreEM, 'r--', linewidth=3 , markersize=8, label='BIC Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/EMScores.png', bbox_inches='tight')
    plt.show()
    

    print("End of Expectation Maximization Analysis");     
      

###### End of KMeans Clustering Analysis ######
    
###### Start of PCA Analysis ######
    
    
def clusteringAnalysisPCA (x_data, y_data):
    print("Starting  PCA Analysis.....");
    numOfComp=16
    mComponentsPCA=[]    
    varPCA=[]    
    svPCA=[]
           
    pca=PCA(n_components=numOfComp, svd_solver='full')
    pca.fit(x_data)
    ev=pca.explained_variance_ratio_
    varPCA.append(ev)
    sv=pca.singular_values_
    svPCA.append(sv)
    mComponentsPCA=np.array(range(1,numOfComp+1))
        
    #### KMeans Analysis
    kClusters=[]
    ssError=[]
    hmScoreKM=[]
    cmScoreKM=[]
    amiScoreKM=[]
    reducedComponents=11
    for k in range(1, 50):
        
        reducedData = PCA(n_components=reducedComponents).fit_transform(x_data)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(reducedData)
        
        #Y_labels[]= kmeans.labels_
        homogeneityScore=homogeneity_score(y_data, kmeans.labels_)
        hmScoreKM.append(homogeneityScore)
        
        completenessScore=completeness_score(y_data, kmeans.labels_)
        cmScoreKM.append(completenessScore)
        
        adjustedMIScore=adjusted_mutual_info_score(y_data, kmeans.labels_)
        amiScoreKM.append(adjustedMIScore)
        
        #sse[k] = kmeans.inertia_
        ssError.append(kmeans.inertia_)
        kClusters.append(k)
    
    
    #### EM Analysis
    mComponents=[]    
    scoreSamplesEM=[]    
    aicSCoreEM=[]
    bicSCoreEM=[]
    for k in range(1, 50):
        
        reducedData = PCA(n_components=reducedComponents).fit_transform(x_data)
        ##covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
        gmmEM = mixture.GaussianMixture(n_components=k, covariance_type='full').fit(reducedData)
        
        scoreSamples=gmmEM.score(reducedData)
        scoreSamplesEM.append(scoreSamples)
        
        aicScore=gmmEM.aic(reducedData)
        aicSCoreEM.append(aicScore)
        
        bicScore=gmmEM.bic(reducedData)
        bicSCoreEM.append(bicScore)
                
        mComponents.append(k)
    
    dir = os.path.join("./","Dataset02_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)

    
    plt.figure('PCA_01')
    fig, ax1 = plt.subplots()
    plt.title('PCA: Variance & Singular Values', fontsize=16);
    ax1.set_ylabel('Variance', fontsize=15);
    ax1.set_xlabel('No. of Components', fontsize=15);
    ax1.bar(mComponentsPCA, ev, label='Variance')
    #ax1.legend(loc='best', fontsize=12)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Singular Values', fontsize=15);
    ax2.set_xlabel('No. of Components', fontsize=15);
    #ax2.set_ylim(0, 950000)
    #ax2.set_yscale('log')
    ax2.plot(mComponentsPCA, sv, 'rs-', linewidth=3 , markersize=8, label='Singular Values')
    ax1.plot(np.nan, 'rs-', label = 'Singular Values')
    ax1.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/evsvPCA.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure('PCAKM_01')
    plt.title('SSE vs Number of Clusters', fontsize=16);
    plt.ylabel('SSE', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, ssError, 'go--', linewidth=3, markersize=8, label='SS Error')
   # plt.plot(kClusters, testAccuracyCTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/PCAKMeansSSE.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure('PCAKM_02')
    plt.title('Scores vs Number of Clusters', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(kClusters, cmScoreKM, 'b*--', linewidth=3 , markersize=8, label='Completeness Score')
    plt.plot(kClusters, amiScoreKM, 'r--', linewidth=3 , markersize=8, label='AMI Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/PCAKMeansScores.png', bbox_inches='tight')
    plt.show()
    
    
    
    plt.figure('PCAEM_01')
    plt.title('Average Log Likelihood vs Number of Components', fontsize=16);
    plt.ylabel('Avg. Log Likelihood', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, scoreSamplesEM, 'go-', linewidth=3 , markersize=8, label='Avg. Log Likelihood')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/PCAEMProbability.png', bbox_inches='tight')
    plt.show(block=False)
    
    plt.figure('PCAEM_02')
    plt.title('Scores vs Number of Components', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, aicSCoreEM, 'bs-', linewidth=3 , markersize=8, label='AIC Score')
    plt.plot(mComponents, bicSCoreEM, 'r--', linewidth=3 , markersize=8, label='BIC Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/PCAEMScores.png', bbox_inches='tight')
    plt.show()
    
    

    print("End of PCA Analysis");   
    
    
###### Start of ICA Analysis ######
    
    
def clusteringAnalysisICA (x_data, y_data):
    print("Starting  ICA Analysis.....");
    numOfComp=16
    mComponentsICA=[]    
    kurICA=[]    
    svICA=[]
    
    mComponentsICA=np.array(range(1,numOfComp+1))    
    
    dims = list(np.arange(2,(x_data.shape[1]-1),3))
    dims.append(x_data.shape[1])
    ica = FastICA(random_state=100)

    for i in range(1, numOfComp+1):
        ica.set_params(n_components=i)
        ica_tmp = ica.fit_transform(x_data)
        ica_tmp = pd.DataFrame(ica_tmp)
        ica_tmp = ica_tmp.kurt(axis=0)
        kurICA.append(ica_tmp.mean())
    
        
    
    #### KMeans Analysis
    kClusters=[]
    ssError=[]
    hmScoreKM=[]
    cmScoreKM=[]
    amiScoreKM=[]
    reducedComponents=13
    for k in range(1, 50):
        
        reducedData = FastICA(n_components=reducedComponents, random_state=100).fit_transform(x_data)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(reducedData)
        
        #Y_labels[]= kmeans.labels_
        homogeneityScore=homogeneity_score(y_data, kmeans.labels_)
        hmScoreKM.append(homogeneityScore)
        
        completenessScore=completeness_score(y_data, kmeans.labels_)
        cmScoreKM.append(completenessScore)
        
        adjustedMIScore=adjusted_mutual_info_score(y_data, kmeans.labels_)
        amiScoreKM.append(adjustedMIScore)
        
        #sse[k] = kmeans.inertia_
        ssError.append(kmeans.inertia_)
        kClusters.append(k)
    
    
    #### EM Analysis
    mComponents=[]    
    scoreSamplesEM=[]    
    aicSCoreEM=[]
    bicSCoreEM=[]
    for k in range(1, 50):
        
        reducedData = FastICA(n_components=reducedComponents, random_state=100).fit_transform(x_data)
        ##covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
        gmmEM = mixture.GaussianMixture(n_components=k, covariance_type='full').fit(reducedData)
        
        scoreSamples=gmmEM.score(reducedData)
        scoreSamplesEM.append(scoreSamples)
        
        aicScore=gmmEM.aic(reducedData)
        aicSCoreEM.append(aicScore)
        
        bicScore=gmmEM.bic(reducedData)
        bicSCoreEM.append(bicScore)
                
        mComponents.append(k)
    
    dir = os.path.join("./","Dataset02_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)

    
    plt.figure('ICA_01')
    fig, ax1 = plt.subplots()
    plt.title('ICA: Kurtosis vs No of Components', fontsize=16);
    ax1.set_ylabel('Kurtosis', fontsize=15);
    ax1.set_xlabel('No. of Independent Components', fontsize=15);
    ax1.bar(mComponentsICA, kurICA, label='Kurtosis')
    #ax1.legend(loc='best', fontsize=12)
   # ax2 = ax1.twinx()
    #ax2.set_ylabel('Singular Values', fontsize=15);
    #ax2.set_xlabel('No. of Independent Components', fontsize=15);
    #ax2.set_ylim(0, 950000)
    #ax2.set_yscale('log')
    #ax2.plot(mComponentsPCA, sv, 'rs-', linewidth=3 , markersize=8, label='Singular Values')
    #ax1.plot(np.nan, 'rs-', label = 'Singular Values')
    ax1.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/kurtICA.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    
    plt.figure('ICAKM_01')
    plt.title('SSE vs Number of Clusters', fontsize=16);
    plt.ylabel('SSE', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, ssError, 'go--', linewidth=3, markersize=8, label='SS Error')
   # plt.plot(kClusters, testAccuracyCTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/ICAKMeansSSE.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure('ICAKM_02')
    plt.title('Scores vs Number of Clusters', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(kClusters, cmScoreKM, 'b*--', linewidth=3 , markersize=8, label='Completeness Score')
    plt.plot(kClusters, amiScoreKM, 'r--', linewidth=3 , markersize=8, label='AMI Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/ICAKMeansScores.png', bbox_inches='tight')
    plt.show()
    
    
    
    plt.figure('ICAEM_01')
    plt.title('Average Log Likelihood vs Number of Components', fontsize=16);
    plt.ylabel('Avg. Log Likelihood', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, scoreSamplesEM, 'go-', linewidth=3 , markersize=8, label='Avg. Log Likelihood')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/ICAEMProbability.png', bbox_inches='tight')
    plt.show(block=False)
    
    plt.figure('ICAEM_02')
    plt.title('Scores vs Number of Components', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, aicSCoreEM, 'bs-', linewidth=3 , markersize=8, label='AIC Score')
    plt.plot(mComponents, bicSCoreEM, 'r--', linewidth=3 , markersize=8, label='BIC Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/ICAEMScores.png', bbox_inches='tight')
    plt.show()
    
    
    

    print("End of ICA Analysis");     
      

###### End of ICA Analysis ######
    
##### Start of RP Analysis #####
    
def clusteringAnalysisRPA (x_data, y_data):
    print("Starting RPA Analysis.....");
    
    originalDist = euclidean_distances(x_data, squared=True).ravel()
    nonzero = originalDist != 0
    originalDist = originalDist[nonzero]
    
    rp=[]
    projected_data=[]
    meanRates=[]
    stdRates=[]
    numOfComp=16
    rand_states=5
    mComponentsRPA=np.array(range(2,numOfComp+1))  
    rStates=np.array(range(rand_states))
    for rnd in rStates:
        print("State", rnd)
        mean=[]
        std=[]
        for n_components in mComponentsRPA:
            print("Component", n_components)
            rp = SparseRandomProjection(n_components=n_components, random_state=rnd)
            projected_data = rp.fit_transform(x_data)
            projectedDist = euclidean_distances(projected_data, squared=True).ravel()[nonzero]
            rates = projectedDist / originalDist
            mean.append(np.mean(rates))
            std.append(np.std(rates))
            
        meanRates.append(mean)
        stdRates.append(std)
          
        
    #print(meanRates)
    
    #### KMeans Analysis
    kClusters=[]
    ssError=[]
    hmScoreKM=[]
    cmScoreKM=[]
    amiScoreKM=[]
    reducedComponents=10
    for k in range(1, 30):
        
        reducedData = SparseRandomProjection(n_components=reducedComponents, random_state=rnd).fit_transform(x_data)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(reducedData)
        
        #Y_labels[]= kmeans.labels_
        homogeneityScore=homogeneity_score(y_data, kmeans.labels_)
        hmScoreKM.append(homogeneityScore)
        
        completenessScore=completeness_score(y_data, kmeans.labels_)
        cmScoreKM.append(completenessScore)
        
        adjustedMIScore=adjusted_mutual_info_score(y_data, kmeans.labels_)
        amiScoreKM.append(adjustedMIScore)
        
        #sse[k] = kmeans.inertia_
        ssError.append(kmeans.inertia_)
        kClusters.append(k)
    
    
    #### EM Analysis
    mComponents=[]    
    scoreSamplesEM=[]    
    aicSCoreEM=[]
    bicSCoreEM=[]
    for k in range(1, 30):
        
        reducedData = FastICA(n_components=reducedComponents, random_state=100).fit_transform(x_data)
        ##covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
        gmmEM = mixture.GaussianMixture(n_components=k, covariance_type='full').fit(reducedData)
        
        scoreSamples=gmmEM.score(reducedData)
        scoreSamplesEM.append(scoreSamples)
        
        aicScore=gmmEM.aic(reducedData)
        aicSCoreEM.append(aicScore)
        
        bicScore=gmmEM.bic(reducedData)
        bicSCoreEM.append(bicScore)
                
        mComponents.append(k)
    
    dir = os.path.join("./","Dataset02_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)

    
    
    plt.figure('RPA_01')
    plt.title('Pairwise squarred distance rates: Mean', fontsize=16);
    plt.ylabel('Mean', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    plt.ylim(0.5, 2.0)
    plt.plot(mComponentsRPA, meanRates[0], 'go-', linewidth=3, markersize=8, label='Mean(rand=1)')
    plt.plot(mComponentsRPA, meanRates[1], 'b*-', linewidth=3, markersize=8, label='Mean(rand=2)')
    plt.plot(mComponentsRPA, meanRates[2], 'rs-', linewidth=3, markersize=8, label='Mean(rand=3)')
    plt.plot(mComponentsRPA, meanRates[3], 'cd-', linewidth=3, markersize=8, label='Mean(rand=4)')
    plt.plot(mComponentsRPA, meanRates[4], 'mx-', linewidth=3, markersize=8, label='Mean(rand=5)')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/meanRPA.png', bbox_inches='tight')
    plt.show(block=False)
    
    plt.figure('RPA_02')
    plt.title('Pairwise squarred distance rates: Std', fontsize=16);
    plt.ylabel('Standard Deviation', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(mComponentsRPA, stdRates[0], 'go--', linewidth=3, markersize=8, label='Std(rand=1)')
    plt.plot(mComponentsRPA, stdRates[1], 'b*--', linewidth=3, markersize=8, label='Std(rand=2)')
    plt.plot(mComponentsRPA, stdRates[2], 'rs--', linewidth=3, markersize=8, label='std(rand=3)')
    plt.plot(mComponentsRPA, stdRates[3], 'cd--', linewidth=3, markersize=8, label='std(rand=4)')
    plt.plot(mComponentsRPA, stdRates[4], 'mx--', linewidth=3, markersize=8, label='std(rand=5)')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/stdRPA.png', bbox_inches='tight')
    plt.show()
    
    
    plt.figure('RPAKM_01')
    plt.title('SSE vs Number of Clusters', fontsize=16);
    plt.ylabel('SSE', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, ssError, 'go--', linewidth=3, markersize=8, label='SS Error')
   # plt.plot(kClusters, testAccuracyCTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/RPAKMeansSSE.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure('RPAKM_02')
    plt.title('Scores vs Number of Clusters', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(kClusters, cmScoreKM, 'b*--', linewidth=3 , markersize=8, label='Completeness Score')
    plt.plot(kClusters, amiScoreKM, 'r--', linewidth=3 , markersize=8, label='AMI Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/RPAKMeansScores.png', bbox_inches='tight')
    plt.show()
    
    
    
    plt.figure('RPAEM_01')
    plt.title('Average Log Likelihood vs Number of Components', fontsize=16);
    plt.ylabel('Avg. Log Likelihood', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, scoreSamplesEM, 'go-', linewidth=3 , markersize=8, label='Avg. Log Likelihood')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/RPAEMProbability.png', bbox_inches='tight')
    plt.show(block=False)
    
    plt.figure('LDAEM_02')
    plt.title('Scores vs Number of Components', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, aicSCoreEM, 'bs-', linewidth=3 , markersize=8, label='AIC Score')
    plt.plot(mComponents, bicSCoreEM, 'r--', linewidth=3 , markersize=8, label='BIC Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/RPAEMScores.png', bbox_inches='tight')
    plt.show()
    
    print("End of RPA Analysis.....");

##### End of RP Analysis #######

###### Start of LDA Analysis ######
    
    
def clusteringAnalysisLDA (x_data, y_data):
    print("Starting  LDA Analysis.....");
    numOfComp=16
    mComponentsLDA=[]    
    varLDA=[]    
    svLDA=[]
    mComponentsLDA=np.array(range(1,numOfComp+1))
            
    lda=LinearDiscriminantAnalysis(n_components=numOfComp, solver='eigen')
    lda.fit(x_data, y_data)
    ev=lda.explained_variance_ratio_
    varLDA.append(ev)            
    sv=np.cumsum(ev)
    svLDA.append(sv)    

    #### KMeans Analysis
    
    kClusters=[]
    ssError=[]
    hmScoreKM=[]
    cmScoreKM=[]
    amiScoreKM=[]
    reducedComponents=10
    for k in range(1, 50):
        
        reducedData = LinearDiscriminantAnalysis(n_components=reducedComponents, solver='eigen').fit_transform(x_data, y_data)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(reducedData)
        
        #Y_labels[]= kmeans.labels_
        homogeneityScore=homogeneity_score(y_data, kmeans.labels_)
        hmScoreKM.append(homogeneityScore)
        
        completenessScore=completeness_score(y_data, kmeans.labels_)
        cmScoreKM.append(completenessScore)
        
        adjustedMIScore=adjusted_mutual_info_score(y_data, kmeans.labels_)
        amiScoreKM.append(adjustedMIScore)
        
        #sse[k] = kmeans.inertia_
        ssError.append(kmeans.inertia_)
        kClusters.append(k)
    
    
    #### EM Analysis
    mComponents=[]    
    scoreSamplesEM=[]    
    aicSCoreEM=[]
    bicSCoreEM=[]
    for k in range(1, 50):
        
        reducedData = LinearDiscriminantAnalysis(n_components=reducedComponents, solver='eigen').fit_transform(x_data, y_data)
        ##covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
        gmmEM = mixture.GaussianMixture(n_components=k, covariance_type='full').fit(reducedData)
        
        scoreSamples=gmmEM.score(reducedData)
        scoreSamplesEM.append(scoreSamples)
        
        aicScore=gmmEM.aic(reducedData)
        aicSCoreEM.append(aicScore)
        
        bicScore=gmmEM.bic(reducedData)
        bicSCoreEM.append(bicScore)
                
        mComponents.append(k)
    
    dir = os.path.join("./","Dataset02_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    
    plt.figure('LDA_01')
    fig, ax1 = plt.subplots()
    plt.title('LDA: Eigen Value & Cumulitive Vriance', fontsize=16);
    ax1.set_ylabel('Eigen Value', fontsize=15);
    ax1.set_xlabel('No. of Components', fontsize=15);
    ax1.bar(mComponentsLDA, varLDA[0], label='Variance')
    #ax1.legend(loc='best', fontsize=12)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulitive Vriance', fontsize=15);
    ax2.set_xlabel('No. of Components', fontsize=15);
    ax2.set_ylim(0, 1)
    #ax2.set_yscale('log')
    ax2.plot(mComponentsLDA, svLDA[0], 'rs-', linewidth=3 , markersize=8, label='Singular Values')
    ax1.plot(np.nan, 'rs-', label = 'Singular Values')
    ax1.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/evsvLDA.png', bbox_inches='tight')
    plt.show(block=False)
    
        
    plt.figure('LDAKM_01')
    plt.title('SSE vs Number of Clusters', fontsize=16);
    plt.ylabel('SSE', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, ssError, 'go--', linewidth=3, markersize=8, label='SS Error')
   # plt.plot(kClusters, testAccuracyCTuned, 'rs' , label='Test Accuracy')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/LDAKMeansSSE.png', bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure('LDAKM_02')
    plt.title('Scores vs Number of Clusters', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Clusters', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
    plt.plot(kClusters, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(kClusters, cmScoreKM, 'b*--', linewidth=3 , markersize=8, label='Completeness Score')
    plt.plot(kClusters, amiScoreKM, 'r--', linewidth=3 , markersize=8, label='AMI Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/LDAKMeansScores.png', bbox_inches='tight')
    plt.show()
    
    
    
    plt.figure('LDAEM_01')
    plt.title('Average Log Likelihood vs Number of Components', fontsize=16);
    plt.ylabel('Avg. Log Likelihood', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, scoreSamplesEM, 'go-', linewidth=3 , markersize=8, label='Avg. Log Likelihood')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/LDAEMProbability.png', bbox_inches='tight')
    plt.show(block=False)
    
    plt.figure('LDAEM_02')
    plt.title('Scores vs Number of Components', fontsize=16);
    plt.ylabel('Score', fontsize=15);
    plt.xlabel('No. of Components', fontsize=15);
    #plt.xlim(0, 160)
    #plt.ylim(50, 70)
#    plt.plot(mComponents, hmScoreKM, 'go--', linewidth=3, markersize=8, label='Homogeneity Score')
    plt.plot(mComponents, aicSCoreEM, 'bs-', linewidth=3 , markersize=8, label='AIC Score')
    plt.plot(mComponents, bicSCoreEM, 'r--', linewidth=3 , markersize=8, label='BIC Score')
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./Dataset02_Plots/LDAEMScores.png', bbox_inches='tight')
    plt.show()

    print("End of LDA Analysis");     
      

###### End of LDA Analysis ######

# ######################################## Neural Network (MLP Classifier) Analysis ############################################

def nnMlpAnalysis (x_data, y_data):
    print("Starting  Neural Network (Multi-layer Perceptron Classifier) Analysis.....");

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    
    reducedTrainPCA = PCA(n_components=11).fit_transform(x_train)
    reducedTrainICA = FastICA(n_components=13).fit_transform(x_train)
    reducedTrainRPA = SparseRandomProjection(n_components=10).fit_transform(x_train)
    reducedTrainLDA = LinearDiscriminantAnalysis(n_components=10, solver='eigen').fit_transform(x_train, y_train)

    
    
    mlp=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)

    start_time = time.time()
    mlp.fit(x_train, y_train)
    fit_time_mlp=time.time()-start_time
    
    y_predict_train=mlp.predict(x_train)
    y_predict_test=mlp.predict(x_test)
    train_mlp=accuracy_score(y_train,y_predict_train.round())
    test_mlp=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlp = cross_val_score(mlp, x_train, y_train, cv=10)
    
    print("performance parameters: NN (MLP) Original")
    print("Fit Time", fit_time_mlp)
    print("Train Score", train_mlp)
    print("Test Score", test_mlp)
    print("CV10 Train Score", scoreCV_mlp.mean())
    
    
    mlpPCA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpPCA.fit(reducedTrainPCA , y_train)
    fit_time_mlpPCA=time.time()-start_time
    
    y_predict_train=mlpPCA.predict(reducedTrainPCA )
    #y_predict_test=mlpPCA.predict(x_test)
    train_mlpPCA=accuracy_score(y_train,y_predict_train.round())
    #test_mlpPCA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpPCA = cross_val_score(mlpPCA, reducedTrainPCA, y_train, cv=10)
    
    print("performance parameters: NN (MLP) Original with PCA")
    print("Fit Time", fit_time_mlpPCA)
    print("Train Score", train_mlpPCA)
    #print("Test Score", test_mlpPCA)
    print("CV10 Train Score", scoreCV_mlpPCA.mean())
    
    
    mlpICA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpICA.fit(reducedTrainICA , y_train)
    fit_time_mlpICA=time.time()-start_time
    
    y_predict_train=mlpICA.predict(reducedTrainICA )
    #y_predict_test=mlpICA.predict(x_test)
    train_mlpICA=accuracy_score(y_train,y_predict_train.round())
    # test_mlpICA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpICA = cross_val_score(mlpICA, reducedTrainICA, y_train, cv=10)
    
    
    print("performance parameters: NN (MLP) Original with ICA")
    print("Fit Time", fit_time_mlpICA)
    print("Train Score", train_mlpICA)
    #print("Test Score", test_mlpICA)
    print("CV10 Train Score", scoreCV_mlpICA.mean())
    
    
    mlpRPA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpRPA.fit(reducedTrainRPA , y_train)
    fit_time_mlpRPA=time.time()-start_time
    
    y_predict_train=mlpRPA.predict(reducedTrainRPA )
    # y_predict_test=mlpRPA.predict(x_test)
    train_mlpRPA=accuracy_score(y_train,y_predict_train.round())
    #test_mlpRPA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpRPA = cross_val_score(mlpRPA, reducedTrainRPA, y_train, cv=10)
    

    print("performance parameters: NN (MLP) Original with RPA")
    print("Fit Time", fit_time_mlpRPA)
    print("Train Score", train_mlpRPA)
    #print("Test Score", test_mlpRPA)
    print("CV10 Train Score", scoreCV_mlpRPA.mean())
    
    
    mlpLDA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpLDA.fit(reducedTrainLDA , y_train)
    fit_time_mlpLDA=time.time()-start_time
    
    y_predict_train=mlpLDA.predict(reducedTrainLDA )
    #y_predict_test=mlpLDA.predict(x_test)
    train_mlpLDA=accuracy_score(y_train,y_predict_train.round())
    #test_mlpLDA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpLDA = cross_val_score(mlpLDA, reducedTrainLDA, y_train, cv=10)
    
    
    print("performance parameters: NN (MLP) Original with LDA")
    print("Fit Time", fit_time_mlpLDA)
    print("Train Score", train_mlpLDA)
    #print("Test Score", test_mlpLDA)
    print("CV10 Train Score", scoreCV_mlpLDA.mean())
    
    
    print("End of  Neural Network (Multi-layer Perceptron Classifier) Analysis");
    
    
      

###### End of Neural Network  Analysis ######
    
    
    
# ######################################## Neural Network (MLP Classifier) Analysis ############################################

def nnMlpAnalysisWithClusters (x_data, y_data):
    print("Starting  Neural Network (Multi-layer Perceptron Classifier) Analysis.....");

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)
    
    kmeans = KMeans(n_clusters=26).fit(x_train)
    kmLabels= kmeans.labels_
    gmmEM = mixture.GaussianMixture(n_components=26, covariance_type='full').fit(x_train)
    emLabels= gmmEM.predict(x_train)
    
    x_train ['KM'] = kmLabels
    x_train ['EM'] = emLabels
    
    x_trainCluster=x_train
    
    reducedTrainPCA = PCA(n_components=11+2).fit_transform(x_trainCluster)
    reducedTrainICA = FastICA(n_components=13+2).fit_transform(x_trainCluster)
    reducedTrainRPA = SparseRandomProjection(n_components=10+2).fit_transform(x_trainCluster)
    reducedTrainLDA = LinearDiscriminantAnalysis(n_components=10+2, solver='eigen').fit_transform(x_trainCluster, y_train)

#    reducedTrainPCA ['KM'] = kmLabels
#    reducedTrainPCA ['EM'] = emLabels
#    
#    reducedTrainICA ['KM'] = kmLabels
#    reducedTrainICA ['EM'] = emLabels
#    
#    reducedTrainRPA ['KM'] = kmLabels
#    reducedTrainRPA ['EM'] = emLabels
#    
#    reducedTrainLDA ['KM'] = kmLabels
#    reducedTrainLDA ['EM'] = emLabels
    
    
    mlp=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)

    start_time = time.time()
    mlp.fit(x_trainCluster, y_train)
    fit_time_mlp=time.time()-start_time
    
    y_predict_train=mlp.predict(x_trainCluster)
    #y_predict_test=mlp.predict(x_test)
    train_mlp=accuracy_score(y_train,y_predict_train.round())
    #test_mlp=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlp = cross_val_score(mlp, x_trainCluster, y_train, cv=10)
    
    print("performance parameters: NN (MLP) Original")
    print("Fit Time", fit_time_mlp)
    print("Train Score", train_mlp)
    #print("Test Score", test_mlp)
    print("CV10 Train Score", scoreCV_mlp.mean())
    
    
    mlpPCA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpPCA.fit(reducedTrainPCA , y_train)
    fit_time_mlpPCA=time.time()-start_time
    
    y_predict_train=mlpPCA.predict(reducedTrainPCA )
    #y_predict_test=mlpPCA.predict(x_test)
    train_mlpPCA=accuracy_score(y_train,y_predict_train.round())
    #test_mlpPCA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpPCA = cross_val_score(mlpPCA, reducedTrainPCA, y_train, cv=10)
    
    print("performance parameters: NN (MLP) Original with PCA")
    print("Fit Time", fit_time_mlpPCA)
    print("Train Score", train_mlpPCA)
    #print("Test Score", test_mlpPCA)
    print("CV10 Train Score", scoreCV_mlpPCA.mean())
    
    
    mlpICA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpICA.fit(reducedTrainICA , y_train)
    fit_time_mlpICA=time.time()-start_time
    
    y_predict_train=mlpICA.predict(reducedTrainICA )
    #y_predict_test=mlpICA.predict(x_test)
    train_mlpICA=accuracy_score(y_train,y_predict_train.round())
    # test_mlpICA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpICA = cross_val_score(mlpICA, reducedTrainICA, y_train, cv=10)
    
    
    print("performance parameters: NN (MLP) Original with ICA")
    print("Fit Time", fit_time_mlpICA)
    print("Train Score", train_mlpICA)
    #print("Test Score", test_mlpICA)
    print("CV10 Train Score", scoreCV_mlpICA.mean())
    
    
    mlpRPA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpRPA.fit(reducedTrainRPA , y_train)
    fit_time_mlpRPA=time.time()-start_time
    
    y_predict_train=mlpRPA.predict(reducedTrainRPA )
    # y_predict_test=mlpRPA.predict(x_test)
    train_mlpRPA=accuracy_score(y_train,y_predict_train.round())
    #test_mlpRPA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpRPA = cross_val_score(mlpRPA, reducedTrainRPA, y_train, cv=10)
    

    print("performance parameters: NN (MLP) Original with RPA")
    print("Fit Time", fit_time_mlpRPA)
    print("Train Score", train_mlpRPA)
    #print("Test Score", test_mlpRPA)
    print("CV10 Train Score", scoreCV_mlpRPA.mean())
    
    
    mlpLDA=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  n_iter_no_change=25, nesterovs_momentum=True, power_t=0.5,
                  random_state=100, shuffle=True, solver='lbfgs', tol=1e-05,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    
    
    start_time = time.time()
    mlpLDA.fit(reducedTrainLDA , y_train)
    fit_time_mlpLDA=time.time()-start_time
    
    y_predict_train=mlpLDA.predict(reducedTrainLDA )
    #y_predict_test=mlpLDA.predict(x_test)
    train_mlpLDA=accuracy_score(y_train,y_predict_train.round())
    #test_mlpLDA=accuracy_score(y_test,y_predict_test.round())
    scoreCV_mlpLDA = cross_val_score(mlpLDA, reducedTrainLDA, y_train, cv=10)
    
    
    print("performance parameters: NN (MLP) Original with LDA")
    print("Fit Time", fit_time_mlpLDA)
    print("Train Score", train_mlpLDA)
    #print("Test Score", test_mlpLDA)
    print("CV10 Train Score", scoreCV_mlpLDA.mean())
    
    
    print("End of  Neural Network (Multi-layer Perceptron Classifier) Analysis");
    
    
      

###### End of Neural Network  Analysis with Clusters ######

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
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=100)

if len(sys.argv)<2:
    print("No options specified.\nDefault option: KMeans Clustering Algorithm")
    clusteringAnalysisKMeans (x_data, y_data)   
elif sys.argv[1]=='KM':
    clusteringAnalysisKMeans (x_data, y_data)
elif sys.argv[1]=='EM':
    clusteringAnalysisEMaximization (x_data, y_data)
elif sys.argv[1]=='PCA':
    clusteringAnalysisPCA (x_data, y_data)
elif sys.argv[1]=='ICA':
    clusteringAnalysisICA (x_data, y_data)
elif sys.argv[1]=='RPA':
    clusteringAnalysisRPA (x_data, y_data)
elif sys.argv[1]=='LDA':
    clusteringAnalysisLDA (x_data, y_data)
elif sys.argv[1]=='NNA':    
    nnMlpAnalysis (x_data, y_data)
elif sys.argv[1]=='NNACluster':    
    nnMlpAnalysisWithClusters(x_data, y_data)
else:
    print("Invalid command-line argument, Valid options are: KMC (default), ADB, SVM, KNN, NNC.")
