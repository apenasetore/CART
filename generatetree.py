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

def maketree(train):
    model = tree.DecisionTreeClassifier(criterion=criterion,
                                        splitter=splitter,
                                        max_depth=mdepth,
                                        class_weight=clweight,
                                        min_samples_leaf=minleaf,
                                        random_state=0,
                                        )      
    
    clf = model.fit(X_train, y_train)