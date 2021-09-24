from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

      
def find_best_split(X, y):
#       Find the best question to ask by iterating over every feature / value and calculating the information gain.
#        best_gain = 0  # keep track of the best information gain
#        best_question = None  # keep track of the feature / value that produced it
#        current_uncertainty = gini(rows)
        best_gain, best_attribute, best_value = 0, 0, 0
        n_features = len(X[0])  # number of columns

        for row in range(len(X)):  # for each feature
    
            for col in range(n_features):  # for each value
    
                
                split_val=X[row][col]
                x_left_temp, x_right_temp, y_left_temp, y_right_temp = partition_classes(X,y,col,split_val)
    
                # Skip this split if it doesn't divide the
                # dataset.
                if len(x_left_temp) == 0 or len(x_right_temp) == 0:
                    continue
    
                # Calculate the information gain from this split
                current_y = [y_left_temp, y_right_temp]
                gain = information_gain(y,current_y)

                if gain >= best_gain:
                    best_gain, best_attribute, best_value = gain, col, split_val

        return best_gain, best_attribute, best_value    

def Predict(y):
    if len(set(y))==1:            
        return y[0]
    else:
        counts = {}  # a dictionary of label -> count.
        for row in y:
            label = row
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return max(counts, key=lambda key: counts[key])
        

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
#        pass
    
    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        
        
#        print((X))
        gain, attribute, value=find_best_split(X,y)
        
        if gain<=0:
            self.tree['label']=Predict(y)
            self.tree['leaf']='Yes'     
            return
        
        x_left, x_right, y_left, y_right = partition_classes(X,y,attribute,value)
        
        self.tree['left'] = DecisionTree()
        self.tree['right']= DecisionTree()
        self.tree['left'].learn(x_left, y_left)
        self.tree['right'].learn(x_right, y_right)
        self.tree['attribute'] = attribute
        self.tree['value'] = value
        self.tree['leaf'] = 'No'
        
#        pass


    def classify(self, record):
        
        # TODO: classify the record using self.tree and return the predicted label
      
        if self.tree['leaf']=='Yes':
            return self.tree['label']
        
        if isinstance(record[self.tree['attribute']],int):
            if record[self.tree['attribute']]<=self.tree['value']:
                return self.tree['left'].classify(record)
            else:
                return self.tree['right'].classify(record)
        else:
            if record[self.tree['attribute']]==self.tree['value']:
                return self.tree['left'].classify(record)
            else:
                return self.tree['right'].classify(record)

       
#        pass
