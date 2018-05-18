from __future__ import division
import numpy
import os
import pandas
import sklearn
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from operator import itemgetter
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt


input_dir = "../CPData/CPOutput/"
output_dir = "MLOutput/"
curr_dir = os.getcwd()

exclude_featuresBF = [0,1,3,4,7,17,19]
exclude_featuresBF.extend(range(70,77))
exclude_featuresDF = range(0,50)
exclude_featuresDF.extend(range(70,77))

os.chdir(input_dir)

#class_names = {'Anaphase','G1','G2','Metaphase','Prophase','S','Telophase'};
#class_labs  = [    4,       1,   1,      3,         2,      1,      5     ];

brightfield_filename = 'BF_cells_on_grid.txt';
darkfield_filename  = 'SSC.txt';

brightfield = pandas.read_csv(brightfield_filename, sep='\t', )
darkfield = pandas.read_csv(darkfield_filename, sep='\t',)

### Preprocessing ###

#exclude features
exclude_featuresBF = [0,1,3,4,7,17,19]
exclude_featuresBF.extend(range(70,77))
exclude_featuresDF = range(0,50)
exclude_featuresDF.extend(range(70,77))

brightfield.drop(brightfield.columns[exclude_featuresBF],axis=1,inplace=True)
darkfield.drop(darkfield.columns[exclude_featuresDF],axis=1,inplace=True)

#build ground truth
ground_truth_list = [4] * 225
ground_truth_list.extend([1] * (103 * 225))
ground_truth_list.extend([3] * 225)
ground_truth_list.extend([2]*(3*225))
ground_truth_list.extend([1] * (39*225))
ground_truth_list.extend([5] * 225)

ground_truth = pandas.DataFrame({'ground_truth': ground_truth_list})


#combine bf and df
data = pandas.concat([brightfield, darkfield, ground_truth], axis=1)

#drop nan
data = data.dropna()

#split data and ground truth
ground_truth = data['ground_truth']
data.drop('ground_truth', axis = 1, inplace=True)

# remove low variance features
selector = VarianceThreshold() #.8 * (1 - .8) 
#print(data.shape)
data = selector.fit_transform(data)
#print(data.shape)

#remove highly correlated features
#skip this for now

os.chdir(curr_dir)
os.chdir(output_dir)


### The plot function ###
def plot(classifier_name, cm_diag):
    bar_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    x_pos = list(range(len(bar_labels)))

    plt.bar(x_pos,
        cm_diag,
        align='center',
        color='blue')

    #plt.grid()

    max_y = 1 #max(cm_diag) 
    plt.ylim([0, 1*1.1])

    plt.ylabel('Percent of cells correctly classifed')
    plt.xticks(x_pos, bar_labels)
    plt.title('Cell Classes, ' + classifier_name)
    plt_name = classifier_name + '_plt.png'
    plt.savefig(plt_name)

### Machine Learning ###
names_classifiers = []
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('AdaBoost', AdaBoostClassifier()))

for name, classifier in names_classifiers:
    #cross validation
    y_pred = sklearn.model_selection.cross_val_predict(classifier, data, ground_truth, cv=10)
    cm = confusion_matrix(ground_truth, y_pred, labels = [1,2,3,4,5])
    #normalize confusion matrix
    row_sums = cm.sum(axis=1)
    normalized_cm = cm / row_sums[:, numpy.newaxis]
    numpy.set_printoptions(precision=3)
    cm_diag = normalized_cm.diagonal()
    cm_file_name = name + '.txt'
    cm_file = open(cm_file_name, 'w+')
    cm_file.write(str(normalized_cm))
    cm_file.close()
    #print(normalized_cm)
    plot(name, cm_diag)




