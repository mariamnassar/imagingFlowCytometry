#In this script, the classifiers are trained and saved as pickle files#

from __future__ import division
import numpy
import os
import pandas
import sklearn
import sys
import getopt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from operator import itemgetter
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import pickle

cell_types = ["B", "T", "eosinophil", "monocyte", "neutrophil"]
train_data_dir = "/users/stud00/mn260/CS/ifc/working_dir/Cell_Profiler_Data/CP_Features_White_Cells"	#the folder which contains the folders with the CP output for each patient id in patient_ids
output_dir = "/users/stud00/mn260/CS/ifc/working_dir/Classifiers"

#exclude features
exclude_featuresDF = ['Location_CenterMassIntensity_X_DF_image', 'Location_CenterMassIntensity_Y_DF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_DF_image', 'Location_MaxIntensity_Y_DF_image', 'Number_Object_Number']

exclude_featuresBF = ['ImageNumber', 'ObjectNumber', 'AreaShape_Center_X', 'AreaShape_Center_Y', 'AreaShape_EulerNumber', 'AreaShape_Orientation', 'AreaShape_Solidity', 'Location_CenterMassIntensity_X_BF_image', 'Location_CenterMassIntensity_Y_BF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_BF_image', 'Location_MaxIntensity_Y_BF_image', 'Number_Object_Number']

exclude_featuresDF.extend(exclude_featuresBF)
exclude_features = exclude_featuresDF

curr_dir = os.getcwd()
os.chdir(curr_dir)

### Train Data Reading and Preprocessing ###
data_filename = 'BF_cells_on_grid.txt';
	
cell_types_data = []
for cell_type in cell_types:
	cell_type_data = pandas.read_csv("".join([train_data_dir, "/", cell_type, "/", data_filename]), sep='\t')
	if cell_type in ["B", "T"]:
		ground_truth_list = ["lymphocyte"] * len(cell_type_data)
	else:
		ground_truth_list = [cell_type] * len(cell_type_data)
	cell_type_data = cell_type_data.drop(exclude_features, axis=1)
	cell_type_data = cell_type_data[cell_type_data.columns.drop(list(cell_type_data.filter(regex='Marker_image')))]
	cell_type_data = cell_type_data.assign(ground_truth = ground_truth_list)
	cell_types_data.append(cell_type_data)

train_data = pandas.concat(cell_types_data)
train_data = train_data.dropna()
ground_truth = train_data['ground_truth']
train_data = train_data.drop('ground_truth', axis = 1)
# remove low variance features
#selector = VarianceThreshold(.1)
#columns = train_data.columns
#train_data = selector.fit_transform(train_data)
#labels = [columns[i] for i in selector.get_support(indices=True)]
#train_data = pandas.DataFrame(train_data, columns=labels)
# Features standardization- skip this, it gives worse results
#norm_train_data = pandas.DataFrame(columns=train_data.columns)
#for feature_name in train_data.columns:
#	norm_train_data[feature_name] = (train_data[feature_name] - train_data[feature_name].mean()) / train_data[feature_name].std()
#train_data = norm_train_data

cell_types = ["lymphocyte", "eosinophil", "monocyte", "neutrophil"]

def write_feature_importances_to_file (feature_importances, name):
	feature_importances_file = open('feature_importances_' + name + '.txt', 'w+')
	feature_importances_file.write("Features sorted by their score:\n")
	for tupl in feature_importances:
		feature_importances_file.write(tupl[1] + ":  " + str(tupl[0]) + "\n")
	feature_importances_file.close()
	
    
######### Machine Learning #########
names_classifiers = []
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))
#names_classifiers.append(('AdaBoost', AdaBoostClassifier()))

os.chdir(output_dir)
for name, classifier in names_classifiers:
	print(name + "\n")
	classifier.fit(train_data, ground_truth)
	#best features
	if name != 'NaiveBayes' :
		feature_names = train_data.columns
		feature_importances = sorted(zip(classifier.feature_importances_, feature_names),reverse=True)
		write_feature_importances_to_file(feature_importances, name)
	#save classifier to file
	pkl_filename = name + "_classifier.pkl"
	with open(pkl_filename, 'wb') as file:
		pickle.dump(classifier, file)
			
###################################
