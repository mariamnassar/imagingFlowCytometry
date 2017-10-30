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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from operator import itemgetter
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
    
cell_types = ["B", "T", "eosinophil", "monocyte", "neutrophil"]
input_dir = "../../CPOutputWhiteCells/2014_2015"	#the folder which contains the folders with the CP output for each patient id in patient_ids
output_dir = "MLOutput/"
montages_dir = "../../Montages"

# for each cell_type create an entry (cell_type, list cell_type = [[patient_id1, num of montages for this patient in all batches], [patient_id2, num of montages..], ..]) in the montages_numbers_dic
montages_numbers_dic = dict()
for cell_type in cell_types:
	cell_type_montages_numbers = []
	cell_type_dir = montages_dir + "/" + cell_type
	patients_ids = next(os.walk(cell_type_dir))[1]
	for patient_id in patients_ids:
		patient_id_dir = cell_type_dir + "/" + patient_id
		all_montages_files = [x[2] for x in os.walk(patient_id_dir)]
		num_of_montages = sum([len(l) for l in all_montages_files])/3
		cell_type_montages_numbers.append([patient_id, num_of_montages])	
	montages_numbers_dic[cell_type] = cell_type_montages_numbers

#exclude features
exclude_featuresDF = ['Location_CenterMassIntensity_X_DF_image', 'Location_CenterMassIntensity_Y_DF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_DF_image', 'Location_MaxIntensity_Y_DF_image', 'Number_Object_Number']

exclude_featuresBF = ['ImageNumber', 'ObjectNumber', 'AreaShape_Center_X', 'AreaShape_Center_Y', 'AreaShape_EulerNumber', 'AreaShape_Orientation', 'AreaShape_Solidity', 'Location_CenterMassIntensity_X_BF_image', 'Location_CenterMassIntensity_Y_BF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_BF_image', 'Location_MaxIntensity_Y_BF_image', 'Number_Object_Number']

exclude_featuresDF.extend(exclude_featuresBF)
exclude_features = exclude_featuresDF

curr_dir = os.getcwd()
os.chdir(curr_dir)

data_filename = 'BF_cells_on_grid.txt';

patients_data = []
for cell_type in cell_types:
	patients_data_cell_type = []  # = [B, T ,..] B=[pid1, pid2,...]  pid1 = dataframe with features for patient1, B 
	count = 0
	cell_type_data = pandas.read_csv("".join([input_dir, "/", cell_type, "/", data_filename]), sep='\t')
	ground_truth_list = [cell_type] * len(cell_type_data)
	cell_type_data = cell_type_data.drop(exclude_features, axis=1)
	cell_type_data = cell_type_data.assign(ground_truth = ground_truth_list)
	cell_type_montages_numbers = montages_numbers_dic.get(cell_type)
	for tupl in cell_type_montages_numbers:
		patient_id = tupl[0]
		num_of_montages = tupl[1]
		patient_data = cell_type_data.take(cell_type_data.index[count:count+900*num_of_montages])
		count = count + num_of_montages * 900
		patient_data = patient_data.dropna()
		patients_data_cell_type.append(patient_data)
	patients_data.append(patients_data_cell_type)

num_cells = [[cell_type_data[i].shape[0] for cell_type_data in patients_data] for i in range(0, len(patients_data[0]))]
# num_cells = a list of: for each patients a list with the number of cells for each cell_type
# num_cells =[p1_num_cells, p2_num_cells, ..]: pi_num_cells = [#B, #T, ..]

patients_data = [pandas.concat([cell_type_data[i] for cell_type_data in patients_data]).dropna(axis=1) for i in range(0, len(patients_data[0]))]
# patients_data = [pid1, pid2,..] pid1 the dataframe for patient1 (all cell_types)

patients_ground_truth = [patient_data['ground_truth'] for patient_data in patients_data]
# patients_data = [pid1_ground_truth, pid2_ground_truth,..] 

# remove the ground_truth column
patients_data = [patient_data.drop('ground_truth', axis = 1) for patient_data in patients_data]


os.chdir(output_dir)


### The plot function for the confusion matrix diagonal###
def plot_cm(plot_file_name, cm_diag):
    bar_labels = cell_types
    opacity = 0.8
    y_pos = numpy.arange(len(bar_labels))
    plt.bar(y_pos, cm_diag, alpha=opacity, align='center', color='blue')
    plt.ylabel('Percent of cells correctly classifed')
    plt.xticks(y_pos, bar_labels)
    plt.title(plot_file_name)
    plt_name = plot_file_name + '_plt.png'
    plt.savefig(plt_name)
    plt.clf()


### The plot function for the comparison of the confusion matrix diagonals for some cell type###
def plot_cm_comp(plot_file_name, cm_diags_cell_type, num_cells_cell_type):
	bar_labels = patients_ids
	opacity = 0.8
	y_pos = numpy.arange(len(bar_labels))
	rects = plt.bar(y_pos, cm_diags_cell_type, alpha=opacity, align='center', color='blue')
	plt.ylim(ymax=1.10)
	plt.ylabel('Percent of cells correctly classifed')
	plt.xticks(y_pos, bar_labels)
	plt.title(plot_file_name, y=1.05)
	for j in range(0,len(rects)):
		rect = rects[j]
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width()/2., 1.04*height, num_cells_cell_type[j], ha='center', va='bottom')
	plt_name = plot_file_name + '_plt.png'
	plt.savefig(plt_name)
	plt.clf()


### The plot function for the comparison of the number of cells (patients, average WBC count)###
def plot_wbc_count(plot_file_name, wbc_count_patient, wbc_counts_avg):
	bar_labels = cell_types
	bar_width = 0.35
	opacity = 0.8
	y_pos = numpy.arange(len(bar_labels))
	plt.bar(y_pos, wbc_count_patient, bar_width, alpha=opacity, align='center', color='red', label='patient')
	plt.bar(y_pos + bar_width, wbc_counts_avg, bar_width, alpha=opacity, align='center', color='green', label='average')
	plt.ylabel('Percent of cells')
	plt.xticks(y_pos + bar_width/2, bar_labels)
	plt.ylim(0.0, 1.0)
	plt.title(plot_file_name)
	plt_name = plot_file_name + '_plt.png'
	plt.legend()
	plt.savefig(plt_name)
	plt.clf()


### Machine Learning ###
names_classifiers = []
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('RandomForest', RandomForestClassifier()))
#names_classifiers.append(('AdaBoost', AdaBoostClassifier()))
names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))



for name, classifier in names_classifiers:
	cm_diags_list =[]   #a list of the diagonals of the confusion matrix for each patient [cm_diag_patient1, cm_diag_patient2, ...]
	wbc_counts = [] #a list of the wbc_count for each patient : wbc_count = [#B/sum_cells, #T/sum_cells,..]
	wbc_counts_sum = [0] * len(cell_types)  #a list of the sums of wbc_counts over all patients
	for i in range(0,len(patients_data)):
		test_data = patients_data[i]
		#test_data = pandas.DataFrame(preprocessing.scale(test_data.values, axis=0))
		test_ground_truth = patients_ground_truth[i]
		#combine pandas train frames
		train_data = pandas.concat(patients_data[:i]+patients_data[i+1:])
		#train_data = pandas.DataFrame(preprocessing.scale(train_data.values, axis=0))
		train_ground_truth = pandas.concat(patients_ground_truth[:i]+patients_ground_truth[i+1:])
		classifier.fit(train_data, train_ground_truth)
		prediction = classifier.predict(test_data)
		cm = confusion_matrix(test_ground_truth, prediction, labels = cell_types)
		cm_diag = cm.diagonal()
		sum_cells = sum(cm_diag)
		wbc_count = cm_diag / sum_cells
		wbc_counts.append(wbc_count)
		wbc_counts_sum = [wbc_counts_sum[j] + wbc_count[j] for j in range(0, len(cell_types))]
		#normalize confusion matrix
		row_sums = cm.sum(axis=1)
		normalized_cm = cm / row_sums[:, numpy.newaxis]
		numpy.set_printoptions(precision=3, suppress=True)
		cm_diag = normalized_cm.diagonal()
		cm_file_name = name + str(i+1) + '_cm.txt'
		cm_file = open(cm_file_name, 'w+')
		cm_file.write(str(normalized_cm))
		cm_file.close()
		cm_diags_list.append(cm_diag)
		plot_cm(name + str(i+1), cm_diag)
	avg_cm_diag = []
	standard_deviation_cm_diag = []
	for j in range(0, len(cell_types)):
		cm_diags_cell_type = numpy.array([item[j] for item in cm_diags_list])
		num_cells_cell_type = numpy.array([item[j] for item in num_cells])
		avg_cm_diag.append(numpy.average(cm_diags_cell_type))
		standard_deviation_cm_diag.append(numpy.std(cm_diags_cell_type))
		plot_cm_comp(name + '_' + cell_types[j], cm_diags_cell_type, num_cells_cell_type)
	plot_cm(name + '_avg_cm', avg_cm_diag)
	plot_cm(name + '_standard_deviation_cm', standard_deviation_cm_diag)
	###Plot the wbc_counts###
	wbc_counts_avg = numpy.array(wbc_counts_sum) / len(patients_data)
	for i in range(0,len(patients_data)):
		plot_file_name = name + "_" + str(i) + "_wbc_count"
		plot_wbc_count(plot_file_name, wbc_counts[i], wbc_counts_avg)
		

###Write the number of cells to files###
num_cells_file = open('number_of_cels', 'w+')		
for i in range(0, len(num_cells)):
	num_cells_file.write(patients_ids[i]+ "   ")
	for j in range(0,5):
		num_cells_file.write("        " + cell_types[j]+ ": " + str(num_cells[i][j]))
	num_cells_file.write("\n")
num_cells_file.close() 
