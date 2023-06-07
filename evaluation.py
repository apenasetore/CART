from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn import tree

import plotly.express as px
import plotly.graph_objects as go
import graphviz

def evaluate_tree(model, test_data, res_test_data):

    score_te = model.score(test_data, res_test_data)
    print('Accuracy Score: ', score_te)

    print(classification_report(res_test_data, model.predict(test_data)))

def plot_model(model):

    return