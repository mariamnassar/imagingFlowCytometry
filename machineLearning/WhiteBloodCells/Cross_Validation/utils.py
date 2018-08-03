import matplotlib
import matplotlib.pyplot as plt
import numpy
import os


def write_feature_importances_to_file (feature_importances, i, classifier, output_dir):
	os.chdir(output_dir)
	feature_importances_file = open(classifier + '_feature_importances_' + str(i+1) + '.txt', 'w+')
	feature_importances_file.write("Features sorted by their score:\n")
	for tupl in feature_importances:
		feature_importances_file.write(tupl[1] + ":  " + str(tupl[0]) + "\n")
	feature_importances_file.close()


def write_ith_accuracy_cm_to_file(cm_normalized_rows, i, classifier, output_dir):
	os.chdir(output_dir)
	cm_file_name = classifier + '_cm_accuracy_' + str(i+1) + '.txt'
	numpy.savetxt(''.join(output_dir + '/' + cm_file_name), cm_normalized_rows, fmt='%.3f')
	
	
def write_ith_sensitivity_cm_to_file(cm_normalized_columns, i, classifier, output_dir):
	os.chdir(output_dir)
	cm_file_name = classifier + '_cm_sensitivity_' + str(i+1) + '.txt'
	numpy.savetxt(''.join(output_dir + '/' + cm_file_name), cm_normalized_columns, fmt='%.3f')
	

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
	
def normalize_rows(cm):
	row_sums = cm.sum(axis=1)
	normalized_cm = cm / row_sums[:, numpy.newaxis]
	numpy.set_printoptions(precision=3, suppress=True)
	normalized_cm = normalized_cm * 100
	return(normalized_cm)

def normalize_columns(cm):
	column_sums = cm.sum(axis=0)
	normalized_cm = cm / column_sums
	numpy.set_printoptions(precision=3, suppress=True)
	normalized_cm = normalized_cm * 100
	return(normalized_cm)


def plot_avg_std_accuracy(avg, std, cell_types, balanced, classifier, output_dir):
	os.chdir(output_dir)
	bar_labels = cell_types
	opacity = 0.8
	bar_width = 0.3
	ax = plt.axes()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(0.0, max(105, 5 + max(numpy.add(avg, std))))
	y_pos = numpy.arange(len(bar_labels))
	if balanced == 'balanced':
		plt.bar(y_pos, avg, yerr=std, alpha=opacity, align='center', color='green', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1))
	else:
		plt.bar(y_pos, avg, yerr=std, alpha=opacity, align='center', color='blue', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1))
	plt.ylabel('13-fold cv average accuracy %', fontsize=13)
	plt.xticks(y_pos, bar_labels, fontsize=13)
	plt.yticks(fontsize=13)
	plt.savefig(classifier + '_accuracy_avg_std.png')
	plt.clf()
	

def plot_avg_std_sensitivity(avg, std, cell_types, balanced, classifier, output_dir):
	os.chdir(output_dir)
	bar_labels = cell_types
	opacity = 0.8
	bar_width = 0.3
	ax = plt.axes()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(0.0, max(105, 5 + max(numpy.add(avg, std))))
	y_pos = numpy.arange(len(bar_labels))
	if balanced == 'balanced':
		plt.bar(y_pos, avg, yerr=std, alpha=opacity, align='center', color='darkolivegreen', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1))
	else:
		plt.bar(y_pos, avg, yerr=std, alpha=opacity, align='center', color='navy', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1))
	plt.ylabel('13-fold cv average sensitivity %', fontsize=13)
	plt.xticks(y_pos, bar_labels, fontsize=13)
	plt.yticks(fontsize=13)
	plt.savefig(classifier + '_sensitivity_avg_std.png')
	plt.clf()
	

def plot_ith_accuracy(accuracy, i, cell_types, balanced, classifier, output_dir):
	os.chdir(output_dir)
	bar_labels = cell_types
	opacity = 0.8
	bar_width = 0.3
	ax = plt.axes()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(0.0, 102)
	y_pos = numpy.arange(len(bar_labels))
	if balanced == 'balanced':
		plt.bar(y_pos, accuracy, alpha=opacity, align='center', color='green', width = bar_width)
	else:
		plt.bar(y_pos, accuracy, alpha=opacity, align='center', color='blue', width = bar_width)
	plt.ylabel('Accuracy', fontsize=13)
	plt.xticks(y_pos, bar_labels, fontsize=13)
	plt.yticks(fontsize=13)
	plot_file_name = classifier + '_accuracy_' + str(i+1) + '.png'
	plt.savefig(plot_file_name)
	plt.clf()	
	

def plot_ith_sensitivity(sensitivity, i, cell_types, balanced, classifier, output_dir):
	os.chdir(output_dir)
	bar_labels = cell_types
	opacity = 0.8
	bar_width = 0.3
	ax = plt.axes()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(0.0, 102)
	y_pos = numpy.arange(len(bar_labels))
	if balanced == 'balanced':
		plt.bar(y_pos, sensitivity, alpha=opacity, align='center', color='darkolivegreen', width = bar_width)
	else:
		plt.bar(y_pos, sensitivity, alpha=opacity, align='center', color='navy', width = bar_width)
	plt.ylabel('Sensitivity', fontsize=13)
	plt.xticks(y_pos, bar_labels, fontsize=13)
	plt.yticks(fontsize=13)
	plot_file_name = classifier + '_sensitivity_' + str(i+1) + '.png'
	plt.savefig(plot_file_name)
	plt.clf()
	

def plot_ith_wbc_count(wbc_count_patient, wbc_counts_ref, i, comp_to, cell_types, classifier, output_dir):
	os.chdir(output_dir)
	bar_labels = cell_types
	bar_width = 0.35
	opacity = 0.8
	y_pos = numpy.arange(len(bar_labels))
	plt.bar(y_pos, wbc_count_patient, bar_width, alpha=opacity, align='center', color='red', label='Patient')
	plt.bar(y_pos + bar_width, wbc_counts_ref, bar_width, alpha=opacity, align='center', color='green', label=comp_to.capitalize())
	plt.ylabel('Percent of cells')
	plt.xticks(y_pos + bar_width/2, bar_labels)
	plt.ylim(0.0, 100)
	plot_file_name = classifier + '_wbc_count_' + comp_to + '_' + str(i+1) + '.png'
	plt.legend()
	plt.savefig(plot_file_name)
	plt.clf()
	

def write_accuracy_avg_to_file(accuracy_avg, cell_types, classifier, output_dir):
	os.chdir(output_dir)
	accuracy_avg_file = open(classifier + '_accuracy_avg.txt', 'w+')
	for i in range(0, len(cell_types)):
		accuracy_avg_file.write(cell_types[i] + ': ' + "%.3f" % accuracy_avg[i] + '\n')
	accuracy_avg_file.close()


def write_sensitivity_avg_to_file(sensitivity_avg, cell_types, classifier, output_dir):
	os.chdir(output_dir)
	sensitivity_avg_file = open(classifier + '_sensitivity_avg.txt', 'w+')
	for i in range(0, len(cell_types)):
		sensitivity_avg_file.write(cell_types[i] + ': ' + "%.3f" % sensitivity_avg[i] + '\n')
	sensitivity_avg_file.close()
	
	
def plot_avg_std_accuracy_sensitivity(accuracy_avg, accuracy_std, sensitivity_avg, sensitivity_std, cell_types, balanced, classifier, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.chdir(output_dir)
	bar_labels = cell_types
	opacity = 0.8
	bar_width = 0.3
	ax = plt.axes()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(0.0, max(105, 5 + max([max(numpy.add(accuracy_avg, accuracy_std)), max(numpy.add(sensitivity_avg, sensitivity_std))])))
	y_pos = numpy.arange(len(bar_labels))
	if balanced == 'balanced':
		plt.bar(y_pos, accuracy_avg, yerr=accuracy_std, alpha=opacity, align='center', color='green', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1), label='Accuracy')
		plt.bar(y_pos + bar_width, sensitivity_avg, yerr=sensitivity_std, alpha=opacity, align='center', color='darkolivegreen', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1), label='Sensitivity')
	else:
		plt.bar(y_pos, accuracy_avg, yerr=accuracy_std, alpha=opacity, align='center', color='blue', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1), label='Accuracy')
		plt.bar(y_pos + bar_width, sensitivity_avg, yerr=sensitivity_std, alpha=opacity, align='center', color='navy', width = bar_width, error_kw=dict(ecolor='black', lw=1.5, capsize=2, apthick=1), label='Sensitivity')
	plt.xticks(y_pos + bar_width/2, bar_labels, fontsize=13)
	plt.yticks(fontsize=12)
	plt.ylabel('13-fold cv average %', fontsize=13)
	plt.legend(loc=3)
	plt.savefig(classifier + '_accuracy_sensitivity_avg_std.png')
	plt.clf()
