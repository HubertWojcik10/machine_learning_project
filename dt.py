'''
This is our implementation of a Decision Tree model from scratch.
'''
import numpy as np
import pandas as pd
from load_data import load_data

class Node():
    '''
        A helper-class used to represent a node in the decision tree
    '''
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        '''
            initialize the node of the tree
        '''
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class InformationGain:
    '''
        A helper-class used to calculate the information gain of a split
    '''
    def __init__(self, parent, l_child, r_child, mode='entropy'):
        self.l_child = l_child
        self.r_child = r_child
        self.weight_l, self.weight_r = len(l_child) / len(parent), len(r_child) / len(parent)
        self.mode = mode
        self.parent = parent
        

    def calculate(self):
        '''
            calculate the information gain based on the chosen mode
        '''
        if self.mode=='gini':
            return self.gini(self.parent) - (self.weight_l*self.gini(self.l_child) + self.weight_r*self.gini(self.r_child))
        elif self.mode=='entropy':
            return self.entropy(self.parent) - (self.weight_l*self.entropy(self.l_child) + self.weight_r*self.entropy(self.r_child))


    def entropy(self, y):
        ''' 
            function to compute entropy 
        '''
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini_i = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini_i


class DecisionTreeClassifier():
    '''
        A class used to represent a Decision Tree Classifier
    '''
    def __init__(self, min_samples_split=2, max_depth=5, mode='entropy'):     
        '''   
            initialize the decision tree classifier
        '''
        # initialize the root of the tree 
        self.root = None
        self.mode = mode
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' 
            recursive function to build the tree 
        ''' 
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split['info_gain']>0:
                # recur left
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                # return decision node
                return Node(best_split['feature_index'], best_split['threshold'], 
                            left_subtree, right_subtree, best_split['info_gain'])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' 
            function to find the best split 
        '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float('inf') #the minimum possible value
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_information_gain = InformationGain(y, left_y, right_y, mode=self.mode).calculate()
                    # update the best split if needed
                    if curr_information_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_information_gain
                        max_info_gain = curr_information_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' 
            function to split the data 
        '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
        
    def calculate_leaf_value(self, Y):
        ''' 
            function to compute leaf node 
        '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' 
            function to print the tree 
        '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print('X_'+str(tree.feature_index), '<=', tree.threshold, '?', tree.info_gain)
            print('%sleft:' % (indent), end='')
            self.print_tree(tree.left, indent + indent)
            print('%sright:' % (indent), end='')
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' 
            function to train the tree 
        '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' 
            function to predict new dataset 
        '''
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' 
            function to predict a single data point 
        '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
