import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import utils
from numpy import loadtxt


main_input_dir = '/Machine_Learning_Outputs/White_Blood_Cells/Cross_Validation/'
main_output_dir = '/Machine_Learning_Outputs/White_Blood_Cells/Cross_Validation_Readouts/'

os.chdir(main_input_dir)
dir_names = next(os.walk('.'))[1]
print(dir_names)
for dir_name in dir_names:
	input_dir = main_input_dir + dir_name
	classification_type = dir_name.split('_')[0]
	balanced = dir_name.split('_')[1]
	output_dir = main_output_dir + dir_name
	if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	if classification_type.lower() == 'all':
		cell_types = ['Lymphocyte', 'Eosinophil', 'Monocyte', 'Neutrophil']
	elif classification_type.lower() == 'lymphocytes':
		cell_types = ['B lymphocyte', 'T lymphocyte']

	classifiers = ['AdaBoost', 'NaiveBayes', 'RandomForest', 'SVC', 'GradientBoosting', 'KNN']

	for classifier in classifiers:
		os.chdir(input_dir)
		accuracy_list = []
		sensitivity_list = []
		wbc_counts = []
		i  = 0
		for subject_cm_file in glob.glob(classifier + '_cm*.txt'):
			path = "".join([input_dir, "/", subject_cm_file])
			cm = loadtxt(path, unpack=True)
			#Transpose the cm: loadtxt reads each line as a column!
			cm = cm.transpose()
			cm_normalized_rows = utils.normalize_rows(cm)
			accuracy_list.append(cm_normalized_rows.diagonal())
			utils.write_ith_accuracy_cm_to_file(cm_normalized_rows, i, classifier, output_dir)
			utils.plot_ith_accuracy(cm_normalized_rows.diagonal(), i, cell_types, balanced, classifier, output_dir)
			cm_normalized_columns = utils.normalize_columns(cm)
			utils.write_ith_sensitivity_cm_to_file(cm_normalized_columns, i, classifier, output_dir)
			utils.plot_ith_sensitivity(cm_normalized_columns.diagonal(), i, cell_types, balanced, classifier, output_dir)
			sensitivity_list.append(cm_normalized_columns.diagonal())
			wbc_count = cm.sum(axis=0)
			wbc_count = (wbc_count / wbc_count.sum()) * 100
			wbc_counts.append(wbc_count)		
			i = i+1
		
		accuracy_avg = []
		accuracy_std = []
		sensitivity_avg = []
		sensitivity_std = []
		wbc_count_average = []
		for j in range(0, len(cell_types)):
			accuracy_cell_type = numpy.array([subject_accuracy[j] for subject_accuracy in accuracy_list])
			sensitivity_cell_type = numpy.array([subject_sensitivity[j] for subject_sensitivity in sensitivity_list])
			accuracy_avg.append(numpy.average(accuracy_cell_type))
			accuracy_std.append(numpy.std(accuracy_cell_type))
			sensitivity_avg.append(numpy.average(sensitivity_cell_type))
			sensitivity_std.append(numpy.std(sensitivity_cell_type))
			cell_type_count = numpy.array([subject_cell_count[j] for subject_cell_count in wbc_counts])
			wbc_count_average.append(numpy.average(cell_type_count))
		utils.write_accuracy_avg_to_file(accuracy_avg, cell_types, classifier, output_dir)
		utils.write_sensitivity_avg_to_file(sensitivity_avg, cell_types, classifier, output_dir)
		utils.plot_avg_std_accuracy(accuracy_avg, accuracy_std, cell_types, balanced, classifier, output_dir)
		utils.plot_avg_std_sensitivity(sensitivity_avg, sensitivity_std, cell_types, balanced, classifier, output_dir)
		utils.plot_avg_std_accuracy_sensitivity(accuracy_avg, accuracy_std, sensitivity_avg, sensitivity_std, cell_types, balanced, classifier, output_dir)
		if (classification_type.lower() == 'all' and balanced == 'unbalanced'):
			wbc_counts_ref = [28.5, 2.3, 5.3, 62.4]
			for k in range(0,13):
				utils.plot_ith_wbc_count(wbc_counts[k], wbc_counts_ref, k, 'reference', cell_types, classifier, output_dir)
				utils.plot_ith_wbc_count(wbc_counts[k], wbc_count_average, k, 'average', cell_types, classifier, output_dir)
		
		
		
