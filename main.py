import pandas as pd
import numpy as np
import re 
import dataprocess
import generate_tree
import evaluation
import random 

from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn import tree

import plotly.express as px
import plotly.graph_objects as go
import graphviz


m = dataprocess.data_txt_to_matrix()
(train_data, res_train_data, test_data, res_test_data) = dataprocess.generate_data(m)
print(len(res_train_data))

model = generate_tree.make_tree(train_data, res_train_data)
evaluation.evaluate_tree(model, test_data, res_test_data)

