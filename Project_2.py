import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier

import graphviz
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plt

import mglearn
import graphviz

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import seaborn as sns
sns.set(style="darkgrid")
from IPython.display import display
params = {'legend.fontsize': 16,
          'legend.handlelength': 2,
          'figure.figsize': (12,9),
          'axes.titlesize': 16,
          'axes.labelsize': 16
         }
plt.rcParams.update(params)


import plotly
import plotly.offline as py
from plotly.offline import init_notebook_mode, download_plotlyjs
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)
# plotly.tools.set_credentials_file(username='vkrishnamani', api_key='uTN0DvhXNYXtzrrmwwpG')
cf.set_config_file(offline=True, world_readable=True, theme='pearl')



raw_data = pd.read_csv("Dataset_Copy.csv")

# Create One hot features for the Categorical columns in Dataset
raw_data = pd.get_dummies(raw_data, columns = ['Work class', 'Education', 'MaritalStatus','Occupation','Relationship','Gender'], drop_first=True)


# Separate the Dependent Column
y = raw_data.Income

# Separate the Independent Columns, by dropping Dependent Column
raw_data = raw_data.drop(['Income'], axis=1)


# Final Input and Label datasets using numpy arrays
X = np.array(raw_data)
Y = np.array(y)


# Split data into Train & Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=17)


# Arrays to store the Accuracy for each Classification Model
X_plot = np.array(["Bernoulli NB","Multinomial NB", "Gaussian NB", "Decision Tree", "MultiLayer Perceptron"])
Y_plot = np.array([])



## Bernoulli Naive Bayes Classifier with Binarize

BernNB = BernoulliNB(binarize=0.1)
BernNB.fit(X_train,Y_train)

### Training accuracy
Y_expect = Y_train
Y_pred = BernNB.predict(X_train)
print accuracy_score(Y_expect,Y_pred)

### Testing accuracy
Y_expect = Y_test
Y_pred = BernNB.predict(X_test)
acc = accuracy_score(Y_expect,Y_pred)
print acc

Y_plot = np.append(Y_plot,acc)





## Multinomial Naive Bayes Classifier

MultiNB = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
MultiNB.fit(X_train,Y_train)

### Training accuracy
Y_expect = Y_train
Y_pred = MultiNB.predict(X_train)

print accuracy_score(Y_expect,Y_pred)

### Testing accuracy
Y_expect = Y_test
Y_pred = MultiNB.predict(X_test)

acc = accuracy_score(Y_expect,Y_pred)
print acc

Y_plot = np.append(Y_plot,acc)






## Gaussian Naive Bayes Classifier

GausNB = GaussianNB()
GausNB.fit(X_train,Y_train)

### Training accuracy
Y_expect = Y_train
Y_pred = GausNB.predict(X_train)

print accuracy_score(Y_expect,Y_pred)

### Testing accuracy
Y_expect = Y_test
Y_pred = GausNB.predict(X_test)

acc = accuracy_score(Y_expect,Y_pred)
print acc

Y_plot = np.append(Y_plot,acc)






## Decision tree classifier with Overfitting
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train,Y_train)

### Training accuracy
print tree.score(X_train,Y_train)

### Test accuracy
print tree.score(X_test,Y_test)


features = list(raw_data.columns)
export_graphviz(tree, out_file='census_overfit.dot',class_names=['>=50K','<50K'], feature_names=features, 
                impurity=False, filled=True)





## Decision tree classifier without Overfitting (i.e. less Depth of tree)
tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
tree.fit(X_train,Y_train)

### Training accuracy
print tree.score(X_train,Y_train)

### Test accuracy
acc = tree.score(X_test,Y_test)
print acc

Y_plot = np.append(Y_plot,acc)

features = list(raw_data.columns)
export_graphviz(tree, out_file='census.dot',class_names=['>=50K','<50K'], feature_names=features, 
                impurity=False, filled=True)



### Find important features used in the decision tree
print "Feature  importances {0}".format(tree.feature_importances_)

n_features = len(features)

plt.barh(range(n_features), tree.feature_importances_,align='center')
plt.yticks(np.arange(n_features), features)
plt.xlabel("Feature Importances")
plt.ylabel("Feauture")

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size

plt.show()







# Multi-Layer Perceptron

mlp = MLPClassifier(random_state = 42)
mlp.fit(X_train, Y_train)


### Training accuracy
print  mlp.score(X_train,Y_train)

### Testing accuracy
print mlp.score(X_test,Y_test)




### Feature scaling and lesser hidden layers to prevent Overfitting
scaler = StandardScaler()

X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)


mlp = MLPClassifier(max_iter = 1000, random_state = 42, hidden_layer_sizes=(100,))
mlp.fit(X_train_scaled, Y_train)

### Training accuracy
print mlp.score(X_train_scaled,Y_train)

### Testing accuracy
acc = mlp.score(X_test_scaled,Y_test)
print acc

Y_plot = np.append(Y_plot,acc)

## Check the weights of the perceptron
features = list(raw_data.columns)

plt.figure(figsize=(20,10))
plt.imshow(mlp.coefs_[0], interpolation='None', cmap = 'GnBu')
plt.yticks(range(30), features)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feautre")
plt.colorbar()







# Comparison of accuracies

plt.figure(figsize=(10,5))
plt.ylabel('Accuracies')
plt.plot(X_plot,Y_plot)





# ROC curves - Multilayer Perceptron
Y_pred = mlp.predict(X_test)
y_onehot_test = pd.get_dummies(Y_test, columns = ['Income'])
y_onehot_pred = pd.get_dummies(Y_pred, columns = ['Income'])
y_onehot_test = np.array(y_onehot_test)
y_onehot_pred = np.array(y_onehot_pred)
stats_list = []
for i in range(2):
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(y_onehot_test[:, i], y_onehot_pred[:, i])    
    # Calculate area under the curve
    roc_auc = [auc(fpr, tpr)] * len(fpr)
    classes = [i] * len(fpr)
    stats_list += zip(fpr, tpr, roc_auc, classes)
stats = pd.DataFrame(stats_list, columns=['fpr', 'tpr', 'auc', 'class'])

data = []
for key, grp in stats.groupby(['class']):
    trace = go.Scatter(x = grp.fpr,
                       y = grp.tpr,
                       name = 'class %d' % key)
    data.append(trace)
# Edit the layout
layout = dict(title = 'Receiver Operating Characterstic',
              xaxis = dict(title = 'False Positive Rate',
                           range = [0, 1]),
              yaxis = dict(title = 'True Positive Rate'),
              width=500,
              height=500,
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-line')

stats.groupby('class').mean().auc








# ROC curves - Decision Trees
Y_pred = tree.predict(X_test)
y_onehot_test = pd.get_dummies(Y_test, columns = ['Income'])
y_onehot_pred = pd.get_dummies(Y_pred, columns = ['Income'])
y_onehot_test = np.array(y_onehot_test)
y_onehot_pred = np.array(y_onehot_pred)
stats_list = []
for i in range(2):
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(y_onehot_test[:, i], y_onehot_pred[:, i])    
    # Calculate area under the curve
    roc_auc = [auc(fpr, tpr)] * len(fpr)
    classes = [i] * len(fpr)
    stats_list += zip(fpr, tpr, roc_auc, classes)
stats = pd.DataFrame(stats_list, columns=['fpr', 'tpr', 'auc', 'class'])

data = []
for key, grp in stats.groupby(['class']):
    trace = go.Scatter(x = grp.fpr,
                       y = grp.tpr,
                       name = 'class %d' % key)
    data.append(trace)
# Edit the layout
layout = dict(title = 'Receiver Operating Characterstic',
              xaxis = dict(title = 'False Positive Rate',
                           range = [0, 1]),
              yaxis = dict(title = 'True Positive Rate'),
              width=500,
              height=500,
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-line')

stats.groupby('class').mean().auc







# ROC curves - Naive Bayes

Y_pred = GausNB.predict(X_test)
y_onehot_test = pd.get_dummies(Y_test, columns = ['Income'])
y_onehot_pred = pd.get_dummies(Y_pred, columns = ['Income'])
y_onehot_test = np.array(y_onehot_test)
y_onehot_pred = np.array(y_onehot_pred)
stats_list = []
for i in range(2):
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(y_onehot_test[:, i], y_onehot_pred[:, i])    
    # Calculate area under the curve
    roc_auc = [auc(fpr, tpr)] * len(fpr)
    classes = [i] * len(fpr)
    stats_list += zip(fpr, tpr, roc_auc, classes)
stats = pd.DataFrame(stats_list, columns=['fpr', 'tpr', 'auc', 'class'])

data = []
for key, grp in stats.groupby(['class']):
    trace = go.Scatter(x = grp.fpr,
                       y = grp.tpr,
                       name = 'class %d' % key)
    data.append(trace)
# Edit the layout
layout = dict(title = 'Receiver Operating Characterstic',
              xaxis = dict(title = 'False Positive Rate',
                           range = [0, 1]),
              yaxis = dict(title = 'True Positive Rate'),
              width=500,
              height=500,
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-line')

stats.groupby('class').mean().auc
