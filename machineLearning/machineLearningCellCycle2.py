import numpy
import os
import pandas
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from operator import itemgetter
from collections import Counter

input_dir = "../CPData/CPOutput/"
output_dir = "/MLOutput/"


os.chdir(input_dir)

#class_names = {'Anaphase','G1','G2','Metaphase','Prophase','S','Telophase'};
#class_labs  = [    4,       1,   1,      3,         2,      1,      5     ];

brightfield_filename = 'BF_cells_on_grid.txt';
darkfield_filename  = 'SSC.txt';

brightfield = pandas.read_csv(brightfield_filename, sep='\t',)
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
selector = VarianceThreshold(.8 * (1 - .8))
#print(data.shape)
data = selector.fit_transform(data)
#print(data.shape)

#remove highly correlated features
#TODO


### Machine Learning ###


classifier = RandomForestClassifier()
for i in range(1, 20):
    data_train, data_test, ground_truth_train, ground_truth_test = train_test_split(data, ground_truth, train_size = 0.9)
    classifier.fit(data_train, ground_truth_train)
    y_pred = classifier.predict(data_test)
    cm = confusion_matrix(y_pred, ground_truth_test, labels = [1,2,3,4,5])
    #classify ground_truth_test: how many cells in which class
    num_of_cells_in_classes = Counter(ground_truth_test)
    num_of_cells_in_classes = sorted(num_of_cells_in_classes.items(), key=itemgetter(0))
    print(num_of_cells_in_classes)
    print(cm)


########### Testing ###########
#print(cm)
#print(ground_truth)
#print(brightfield.head(20))

#print(data.head(20))
#print(brightfield.shape)
#print(darkfield.shape)
#print(data.shape)
#print(brightfield['ImageNumber'])
