import pandas as pd
import numpy as np
import re 
import dataprocess
import random 

from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn import tree

import plotly.express as px
import plotly.graph_objects as go
import graphviz


m = dataprocess.data_txt_to_matrix()
ts = dataprocess.split_train_smples(m)
print(len(ts[0]))
print(len(ts[1]))
