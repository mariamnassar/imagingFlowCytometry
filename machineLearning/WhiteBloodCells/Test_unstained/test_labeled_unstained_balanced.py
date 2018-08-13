from __future__ import division
import numpy
import os
import pandas
import sklearn
import sys
import getopt
import pickle
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import utils


cell_types = ['lymphocyte', 'eosinophil', 'monocyte', 'neutrophil']
classifiers_dir = '/Classifiers_balanced/'
test_data_dir = ''
output_dir = ''

#Exclude features
exclude_featuresDF = ['Location_CenterMassIntensity_X_DF_image', 'Location_CenterMassIntensity_Y_DF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_DF_image', 'Location_MaxIntensity_Y_DF_image', 'Number_Object_Number']

exclude_featuresBF = ['ImageNumber', 'ObjectNumber', 'AreaShape_Center_X', 'AreaShape_Center_Y', 'AreaShape_EulerNumber', 'AreaShape_Orientation', 'AreaShape_Solidity', 'Location_CenterMassIntensity_X_BF_image', 'Location_CenterMassIntensity_Y_BF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_BF_image', 'Location_MaxIntensity_Y_BF_image', 'Number_Object_Number']

exclude_featuresDF.extend(exclude_featuresBF)
exclude_features = exclude_featuresDF

curr_dir = os.getcwd()
os.chdir(curr_dir)

#Test data reading and preprocessing
test_subject = os.path.basename(os.path.normpath(test_data_dir))
data_filename = 'BF_cells_on_grid.txt';
cell_types_data = []
for cell_type in cell_types:
	cell_type_data = pandas.read_csv(''.join([test_data_dir, '/', cell_type, '/', data_filename]), sep='\t')
	ground_truth_list = [cell_type] * len(cell_type_data)
	cell_type_data = cell_type_data.drop(exclude_features, axis=1)
	cell_type_data = cell_type_data[cell_type_data.columns.drop(list(cell_type_data.filter(regex='Marker_image')))]
	cell_type_data = cell_type_data.assign(ground_truth = ground_truth_list)
	cell_types_data.append(cell_type_data)
	
test_data = pandas.concat(cell_types_data)
test_data = test_data.dropna()
test_ground_truth = test_data['ground_truth']
test_data = test_data.drop('ground_truth', axis = 1)

    
#Machine Learning
names_classifiers = []
names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))
names_classifiers.append(('KNN', KNeighborsClassifier()))
names_classifiers.append(('SVC', SVC()))
names_classifiers.append(('AdaBoost', AdaBoostClassifier()))


for name, classifier in names_classifiers:
	print(name + '\n')
	#Load trained classifier
	os.chdir(output_dir)
	pkl_filename = name + '_classifier.pkl'
	with open(classifiers_dir + '/' + pkl_filename, 'rb') as file:
		pickle_model = pickle.load(file)
	prediction = pickle_model.predict(test_data)
	wbc_count = Counter(prediction)
	utils.write_wbc_count_to_file(wbc_count, output_dir)
	wbc_counts_ref = [28.5, 2.3, 5.3, 62.4]
	wbc_count_list = []
	for cell_type in cell_types:
		wbc_count_list.append(wbc_count[cell_type])
	plot_wbc_count(wbc_count_list, wbc_counts_ref, 'reference', cell_types, classifier, output_dir)
