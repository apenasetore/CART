import pandas as pd
import numpy as np
import re 
import random 


from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn import tree

import plotly.express as px
import plotly.graph_objects as go
import graphviz

def make_tree(train_data, res_train_data):
    model = tree.DecisionTreeClassifier(criterion="entropy",
                                        splitter="best",
                                        max_depth=5
                                        )      
    
    clf = model.fit(train_data, res_train_data)

    print("\nTREE GENERATED:")
    print('   Classes: ', clf.classes_)
    print('   Tree Depth: ', clf.tree_.max_depth)
    print('   No. of leaves: ', clf.tree_.n_leaves)
    print('   No. of features: ', clf.n_features_in_)
    print("\n")

    return clf
