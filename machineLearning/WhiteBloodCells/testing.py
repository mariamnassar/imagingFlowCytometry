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

cell_types = ["lymphocyte", "eosinophil", "monocyte", "neutrophil"]
classifiers_dir = "/users/stud00/mn260/CS/ifc/working_dir/Classifiers"
### Parse Arguments ###
test = None
output = None

try:
	optlist, args = getopt.getopt(sys.argv[1:], "",  ['test=', 'output='])
except getopt.GetoptError as e:
    print (str(e))
    print("Usage: %s --test test data dir --output output dir" % sys.argv[0])
    sys.exit(2)
    	
for o, a in optlist:
	if o == '--test':
		test_data_dir = a
	elif o == '--output':
		output_dir = a
	else:
		print("Usage: %s --test test data dir --output output dir" % sys.argv[0])

if test_data_dir is None:
	print ("Please enter the test data dir")
if output_dir is None:
	print ("Please enter the output dir")

#exclude features
exclude_featuresDF = ['Location_CenterMassIntensity_X_DF_image', 'Location_CenterMassIntensity_Y_DF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_DF_image', 'Location_MaxIntensity_Y_DF_image', 'Number_Object_Number']

exclude_featuresBF = ['ImageNumber', 'ObjectNumber', 'AreaShape_Center_X', 'AreaShape_Center_Y', 'AreaShape_EulerNumber', 'AreaShape_Orientation', 'AreaShape_Solidity', 'Location_CenterMassIntensity_X_BF_image', 'Location_CenterMassIntensity_Y_BF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_BF_image', 'Location_MaxIntensity_Y_BF_image', 'Number_Object_Number']

exclude_featuresDF.extend(exclude_featuresBF)
exclude_features = exclude_featuresDF

curr_dir = os.getcwd()
os.chdir(curr_dir)

### Test Data Reading and Preprocessing ###
### Test Data Reading and Preprocessing ###
test_subject = os.path.basename(os.path.normpath(test_data_dir))
data_filename = 'BF_cells_on_grid.txt';
cell_types_data = []
for cell_type in cell_types:
	cell_type_data = pandas.read_csv("".join([test_data_dir, "/", cell_type, "/", data_filename]), sep='\t')
	ground_truth_list = [cell_type] * len(cell_type_data)
	cell_type_data = cell_type_data.drop(exclude_features, axis=1)
	cell_type_data = cell_type_data[cell_type_data.columns.drop(list(cell_type_data.filter(regex='Marker_image')))]
	cell_type_data = cell_type_data.assign(ground_truth = ground_truth_list)
	cell_types_data.append(cell_type_data)
	
test_data = pandas.concat(cell_types_data)
test_data = test_data.dropna()
test_ground_truth = test_data['ground_truth']
test_data = test_data.drop('ground_truth', axis = 1)
# remove the features with low variance (also removed from the train data)
#test_data = test_data[labels]
# Features standardization
#norm_test_data = pandas.DataFrame(columns=test_data.columns)
#for feature_name in test_data.columns:
#	norm_test_data[feature_name] = (test_data[feature_name] - test_data[feature_name].mean()) / test_data[feature_name].std()
#test_data = norm_test_data

##################################
######### Plot functions #########

# The plot function for the comparison of the number of cells (patients, usual WBC count)###
def plot_wbc_count(plot_file_name, wbc_count_patient, wbc_count_ref):
	bar_labels = cell_types
	bar_width = 0.35
	opacity = 0.8
	y_pos = numpy.arange(len(bar_labels))
	plt.bar(y_pos, wbc_count_patient, bar_width, alpha=opacity, align='center', color='red', label='patient')
	plt.bar(y_pos + bar_width, wbc_count_ref, bar_width, alpha=opacity, align='center', color='green', label='usual wbc count')
	plt.ylabel('Percent of cells')
	plt.xticks(y_pos + bar_width/2, bar_labels)
	plt.ylim(0, 101)
	plt.title(plot_file_name)
	plt_name = plot_file_name + '_plt.png'
	plt.legend()
	plt.savefig(plt_name)
	plt.clf()  
	
# The plot function for the confusion matrix diagonal
def plot_cm(plot_file_name, cm_diag):
    bar_labels = cell_types
    opacity = 0.8
    plt.ylim(0.0, 1.01)
    y_pos = numpy.arange(len(bar_labels))
    plt.bar(y_pos, cm_diag, alpha=opacity, align='center', color='blue')
    plt.ylabel('Percent of cells correctly classifed')
    plt.xticks(y_pos, bar_labels)
    plt.title(plot_file_name)
    plt_name = plot_file_name + '_plt.png'
    plt.savefig(plt_name)
    plt.clf()
    
######### Machine Learning #########
names_classifiers = []
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))
#names_classifiers.append(('AdaBoost', AdaBoostClassifier()))


for name, classifier in names_classifiers:
	print(name + "\n")
	#load trained classifier
	os.chdir(output_dir)
	pkl_filename = name + "_classifier.pkl"
	with open(classifiers_dir+"/"+pkl_filename, 'rb') as file:
		pickle_model = pickle.load(file)
	wbc_counts = []
	prediction = pickle_model.predict(test_data)
	cm = confusion_matrix(test_ground_truth, prediction, labels = cell_types)
	#normalize confusion matrix
	row_sums = cm.sum(axis=1)
	normalized_cm = cm / row_sums[:, numpy.newaxis]
	numpy.set_printoptions(precision=3, suppress=True)
	cm_diag = normalized_cm.diagonal()
	cm_file_name = name + test_subject + '_cm.txt'
	cm_file = open(cm_file_name, 'w+')
	cm_file.write(str(normalized_cm))
	cm_file.close()
	plot_file_name = name + test_subject + '_cm_plt.txt'
	plot_cm(plot_file_name, cm_diag)
	#wbc count
	dic_cells_count = Counter(prediction)
	wbc_count = []
	for cell_type in cell_types:
		if cell_type in dic_cells_count:
			wbc_count.append(dic_cells_count.get(cell_type))
		else:
			wbc_count.append(0)
	wbc_counts.append(wbc_count)
	print(dic_cells_count)
	print(wbc_count)
	wbc_count = numpy.array(wbc_count)
	sum_cells = sum(wbc_count)
	wbc_count = (wbc_count * 100) / sum_cells
	wbc_count_usual = [28.5, 2.3, 5.3, 62.4]
	plot_wbc_count(name + "_wbc_count_" + test_subject, wbc_count, wbc_count_usual)
			
###################################
