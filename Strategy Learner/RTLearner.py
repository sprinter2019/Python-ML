""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
class RTLearner(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, leaf_size=5, verbose=False):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   	
        self.verbose=verbose
        self.leaf_size = leaf_size	   	 			  		 			     			  	  		 	  	 		 			  		  			
        #pass  # move along, these aren't the drones you're looking for  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def author(self):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: The GT username of the student  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        return "mhassan49"  # replace tb34 with your Georgia Tech username  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def add_evidence(self, data_x, data_y):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Add training data to learner  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param data_x: A set of feature values used to train the learner  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type data_x: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type data_y: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        #print("In RT Learner.......")
        data = np.column_stack((data_x,data_y))
        #print(data)
        self.tree=self.decisionTree(data)
        #print(self.tree)
        

    def decisionTree(self, data):
        trainx = data[:, 0:-1]
        trainy = data[:, -1:]
        #print("trainy.mean()")
        #print(np.median(trainy))

        if trainx.shape[0] <= self.leaf_size:
            #splitVal = trainy.mean()
            #splitVal = np.median(trainy)
            splitVal = np.median(trainy)
            leaf = np.array([[-1, splitVal, np.nan, np.nan]])
            return  leaf

        elif len(np.unique(trainy)) == 1:
            splitVal = trainy.mean()
            #splitVal = np.median(trainy)
            leaf = np.array([[-1, splitVal, np.nan, np.nan]])
            return  leaf

        else:
            splitIndex= self.find_best_split(trainx, trainy)
            splitVal=np.median(trainx[:, splitIndex])
            #splitVal=np.median(trainx[:, splitIndex])
            #print(splitIndex)
            #print(splitVal)
            leftSplit=data[data[:, splitIndex]<=splitVal]
            rightSplit=data[data[:, splitIndex]>splitVal]

            if rightSplit.shape[0]==0:
                leaf=np.array([[-1, trainy.mean(), np.nan, np.nan]])
                #leaf=np.array([[-1, np.median(trainy), np.nan, np.nan]])
                return leaf

            lefttree=self.decisionTree(leftSplit)
            righttree=self.decisionTree(rightSplit)
            root=np.array([[splitIndex, splitVal, 1, lefttree.shape[0]+1]])
            return np.concatenate((root, lefttree, righttree))

    def find_best_split(self, trainx, trainy):
        splitIndex = np.random.randint(0, trainx.shape[1])            

        return splitIndex

    def query(self, points):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Estimate a set of test points given the model we built.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type points: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: The predicted result of the input data according to the trained model  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        predictions = np.empty([points.shape[0], 1], dtype=float)

        for index, point in enumerate(points, 0):
            leaf = False
            iter = 0
            while not leaf:
                temp = int(self.tree[iter, 0])
                if self.tree[iter, 0] != -1:
                    if point[temp] <= self.tree[iter, 1]:
                        iter = iter + int(self.tree[iter, 2])
                    else:
                        iter = iter + int(self.tree[iter, 3])
                else:
                    predictions[index, 0] = self.tree[iter, 1]
                    leaf = True

        #print("Predictions: ")
        #print(predictions)
        return predictions.flatten()
        #return np.asarray(predictions)
          		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print("the secret clue is 'zzyzx'")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
