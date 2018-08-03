import matplotlib
import matplotlib.pyplot as plt
import numpy
import os


def write_feature_importances_to_file (feature_importances, classifier, output_dir):
	os.chdir(output_dir)
	feature_importances_file = open(classifier + '_feature_importances.txt', 'w+')
	feature_importances_file.write("Features sorted by their score:\n")
	for tupl in feature_importances:
		feature_importances_file.write(tupl[1] + ":  " + str(tupl[0]) + "\n")
	feature_importances_file.close()
	

def write_num_cells_to_file(num_cells, subject_ids, cell_types, output_dir):
	os.chdir(output_dir)
	num_cells_file = open('number_of_cels.txt', 'w+')
	for i in range(0, len(num_cells)):
		num_cells_file.write(subject_ids[i]+ "   ")
		for j in range(0,len(cell_types)):
			num_cells_file.write("        " + cell_types[j]+ ": " + str(num_cells[i][j]))
		num_cells_file.write("\n")
	num_cells_file.close()
	
def write_top_10_features_to_file(top_10_features_dic, output_dir):
	os.chdir(output_dir)
	top_10_features_file = open('used_top_10_features.txt', 'w+')
	for feature, score in top_10_features_dic.items():
		top_10_features_file.write(feature + ': ' + str(score) + '\n')
	top_10_features_file.close()
	
def write_zero_features_to_file(zero_features, output_dir):
	os.chdir(output_dir)
	zero_features_file = open('zero_features.txt', 'w+')
	for feature in zero_features:
		zero_features_file.write(feature + '\n')
	zero_features_file.close()
