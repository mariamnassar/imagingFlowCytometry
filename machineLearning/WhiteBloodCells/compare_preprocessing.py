import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import glob
from scipy.stats import ttest_ind


classifier_name = 'NaiveBayes'
classification_types = ['all', 'lymphocytes']
preprocessing_list = ['norm', 'remove_zero_features', 'top_10_features', 'low_var_0.001', 'low_var_0.0001', 'low_var_1e-05']
balanced_params = ['balanced', 'unbalanced']

output_dir = '/Cross_Validation_Readouts/' + classifier_name
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

for classification_type in classification_types:
	for preprocessing in preprocessing_list:
		for balanced in balanced_params:
			if classification_type == 'all':
				cell_types = ['Lymphocyte', 'Eosinophil', 'Monocyte', 'Neutrophil']
			elif classification_type == 'lymphocytes':
				cell_types = ['B lymphocytes', 'T lymphocytes']
			
			if preprocessing=='norm':
				label1='Without normalization'
				label2='With normalization'
			elif preprocessing=='remove_zero_features':
				label1='Without removing zero score features'
				label2='With removing zero score features'
			elif preprocessing=='top_10_features':
				label1='All features'
				label2='Top 10 features'
			elif preprocessing=='low_var_0.001':
				label1='All features'
				label2='Without low var 10^-3'
			elif preprocessing=='low_var_0.0001':
				label1='All features'
				label2='Without low var 10^-4'
			elif preprocessing=='low_var_1e-05':
				label1='All features'
				label2='Without low var 10^-5'

			plt_name = preprocessing + '_' + balanced + '_' + classification_type + '.png'
			if balanced=='balanced':
				accuracy_color='green'
				sensitivity_color='darkolivegreen'
			else:
				accuracy_color='blue'
				sensitivity_color='navy'
				
			input_dir_1 = '/Cross_Validation_Readouts/' + classification_type.capitalize() + '_' + balanced
			input_dir_2 = '/Cross_Validation_Readouts/' + classification_type.capitalize() + '_' + balanced + '_' + preprocessing

			os.chdir(input_dir_1)
			accuracy_diags_list_1 = []
			for patient_cm in glob.glob(classifier_name + '_cm_accuracy*.txt'):
				path = "".join([input_dir_1, "/", patient_cm])
				cm = numpy.loadtxt(path, unpack=True)
				cm = numpy.nan_to_num(cm)
				cm = cm.transpose()
				cm_diag = cm.diagonal()
				accuracy_diags_list_1.append(cm_diag)
				
			sensitivity_diags_list_1 = []
			for patient_cm in glob.glob(classifier_name + '_cm_sensitivity*.txt'):
				path = "".join([input_dir_1, "/", patient_cm])
				cm = numpy.loadtxt(path, unpack=True)
				cm = numpy.nan_to_num(cm)
				cm = cm.transpose()
				cm_diag = cm.diagonal()
				sensitivity_diags_list_1.append(cm_diag)
				
			os.chdir(input_dir_2)
			accuracy_diags_list_2 = []
			for patient_cm in glob.glob(classifier_name + '_cm_accuracy*.txt'):
				path = "".join([input_dir_2, "/", patient_cm])
				cm = numpy.loadtxt(path, unpack=True)
				cm = numpy.nan_to_num(cm)
				cm = cm.transpose()
				cm_diag = cm.diagonal()
				accuracy_diags_list_2.append(cm_diag)
				
			sensitivity_diags_list_2 = []
			for patient_cm in glob.glob(classifier_name + '_cm_sensitivity*.txt'):
				path = "".join([input_dir_2, "/", patient_cm])
				cm = numpy.loadtxt(path, unpack=True)
				cm = numpy.nan_to_num(cm)
				cm = cm.transpose()
				cm_diag = cm.diagonal()
				sensitivity_diags_list_2.append(cm_diag)
				
			accuracy_avg1 = []
			accuracy_avg2 = []
			sensitivity_avg1 = []
			sensitivity_avg2 = []
			accuracy_std1 = []
			accuracy_std2 = []
			sensitivity_std1 = []
			sensitivity_std2 = []
			for i in range(0, len(cell_types)):
				accuracy_avg1.append(numpy.average([accuracy_diag[i] for accuracy_diag in accuracy_diags_list_1]))
				accuracy_avg2.append(numpy.average([accuracy_diag[i] for accuracy_diag in accuracy_diags_list_2]))
				accuracy_std1.append(numpy.std([accuracy_diag[i] for accuracy_diag in accuracy_diags_list_1]))
				accuracy_std2.append(numpy.std([accuracy_diag[i] for accuracy_diag in accuracy_diags_list_2]))
				sensitivity_avg1.append(numpy.average([sensitivity_diag[i] for sensitivity_diag in sensitivity_diags_list_1]))
				sensitivity_avg2.append(numpy.average([sensitivity_diag[i] for sensitivity_diag in sensitivity_diags_list_2]))
				sensitivity_std1.append(numpy.std([sensitivity_diag[i] for sensitivity_diag in sensitivity_diags_list_1]))
				sensitivity_std2.append(numpy.std([sensitivity_diag[i] for sensitivity_diag in sensitivity_diags_list_2]))
			
			p_values_accuracy = []
			p_values_sensitivity = []
			for i in range(0, len(cell_types)):
				t, p = ttest_ind([e[i] for e in accuracy_diags_list_1], [e[i] for e in accuracy_diags_list_2], equal_var=False)
				p_values_accuracy.append(p)
				print("Accuracy ttest_ind:            t = %g  p = %g" % (t, p))
				t, p = ttest_ind([e[i] for e in sensitivity_diags_list_1], [e[i] for e in sensitivity_diags_list_2], equal_var=False)
				p_values_sensitivity.append(p)
				print("Accuracy ttest_ind:            t = %g  p = %g" % (t, p))
			significance_list_accuracy = ['*' if p_value <=0.05 else ' ' for p_value in p_values_accuracy]
			significance_list_sensitivity = ['*' if p_value <=0.05 else ' ' for p_value in p_values_sensitivity]
			
			os.chdir(output_dir)

			
			bar_labels = cell_types
			opacity_1 = 0.8
			opacity_2 = 0.4
			bar_width = 0.3
			y_pos = numpy.arange(len(bar_labels))

			#Plot accuracy
			fig, (ax, ax2) = plt.subplots(2,1, sharex=True)
			rects1 = ax.bar(y_pos + bar_width, accuracy_avg1, bar_width, alpha=opacity_1, align='center', color=accuracy_color, label=label1, yerr=accuracy_std1, error_kw=dict(ecolor='black', lw=1.5, capsize=2, capthick=1))
			rects2 = ax.bar(y_pos + 2 * bar_width, accuracy_avg2, bar_width, alpha=opacity_2, align='center', color=accuracy_color, label=label2, yerr=accuracy_std2, error_kw=dict(ecolor='black', lw=1.5, capsize=2, capthick=1))
			rects1_2 = ax2.bar(y_pos + bar_width, accuracy_avg1, bar_width, alpha=opacity_1, align='center', color=accuracy_color, label=label1)
			rects2_2 = ax2.bar(y_pos + 2 * bar_width, accuracy_avg2, bar_width, alpha=opacity_2, align='center', color=accuracy_color, label=label2)
			ax.set_ylim(int(min([min(numpy.subtract(accuracy_avg1,accuracy_std1)), min(numpy.subtract(accuracy_avg2,accuracy_std2))]))-5, int(max([max(numpy.add(accuracy_avg1,accuracy_std1)), max(numpy.add(accuracy_avg2,accuracy_std2))]))+5)
			ax2.set_ylim(0, 10)
			ax.spines['bottom'].set_visible(False)
			ax2.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax2.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.tick_params(labeltop='off')  #don't put tick labels at the top
			ax.tick_params(bottom='off')  #don't put tick at the bottom
			ax2.xaxis.tick_bottom()
			ax2.get_yaxis().set_ticks([0,10])
			fig.text(0.04, 0.5, '13-fold cv average accuracy %', va='center', rotation='vertical', fontsize=13)
			plt.xticks(y_pos + 1.5*bar_width, bar_labels, fontsize=13)
			plt.legend(loc=10, fontsize=13)
			for j in range(0,len(cell_types)):
				rect1 = rects1[j]
				rect2 = rects2[j]
				height = max([rect1.get_height(), rect2.get_height()])
				ax.text(rect1.get_x() + rect1.get_width(), height*1.04, significance_list_accuracy[j], ha='center')
			for j in range(0,len(cell_types)):
				height = rects1[j].get_height()
				rect = rects1_2[j]
				ax2.text(rect.get_x() + rect.get_width()/2.0, 10, '%.1f' % height, ha='center', va='bottom', fontsize=12, rotation=30)
			for j in range(0,len(cell_types)):
				height = rects2[j].get_height()
				rect = rects2_2[j]
				ax2.text(rect.get_x() + rect.get_width()/2.0, 10, '%.1f' % height, ha='center', va='bottom', fontsize=12, rotation=30)
			plt.savefig('accuracy_' + plt_name)
			plt.clf()
			
			#Plot sensitivity
			fig, (ax, ax2) = plt.subplots(2,1, sharex=True)
			rects1 = ax.bar(y_pos + bar_width, sensitivity_avg1, bar_width, alpha=opacity_1, align='center', color=sensitivity_color, label=label1, yerr=sensitivity_std1, error_kw=dict(ecolor='black', lw=1.5, capsize=2, capthick=1))
			rects2 = ax.bar(y_pos + 2 * bar_width, sensitivity_avg2, bar_width, alpha=opacity_2, align='center', color=sensitivity_color, label=label2, yerr=sensitivity_std2, error_kw=dict(ecolor='black', lw=1.5, capsize=2, capthick=1))
			ax2.bar(y_pos + bar_width, sensitivity_avg1, bar_width, alpha=opacity_1, align='center', color=sensitivity_color, label=label1)
			ax2.bar(y_pos + 2 * bar_width, sensitivity_avg2, bar_width, alpha=opacity_2, align='center', color=sensitivity_color, label=label2)
			ax.set_ylim(int(min([min(numpy.subtract(sensitivity_avg1,sensitivity_std1)), min(numpy.subtract(sensitivity_avg2,sensitivity_std2))]))-5, int(max([max(numpy.add(sensitivity_avg1,sensitivity_std1)), max(numpy.add(sensitivity_avg2,sensitivity_std2))]))+5)
			ax2.set_ylim(0, 10)
			ax.spines['bottom'].set_visible(False)
			ax2.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax2.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.tick_params(labeltop='off')  #don't put tick labels at the top
			ax.tick_params(bottom='off')  #don't put tick at the bottom
			ax2.xaxis.tick_bottom()
			ax2.get_yaxis().set_ticks([0,10])
			fig.text(0.04, 0.5, '13-fold cv average sensitivity %', va='center', rotation='vertical', fontsize=13)
			plt.xticks(y_pos + 1.5*bar_width, bar_labels, fontsize=13)
			plt.legend(loc=10, fontsize=13)
			for j in range(0,len(cell_types)):
				rect1 = rects1[j]
				rect2 = rects2[j]
				height = max([rect1.get_height(), rect2.get_height()])
				ax.text(rect1.get_x() + rect1.get_width(), height*1.04, significance_list_sensitivity[j], ha='center')
			for j in range(0,len(cell_types)):
				height = rects1[j].get_height()
				rect = rects1_2[j]
				ax2.text(rect.get_x() + rect.get_width()/2.0, 10, '%.1f' % height, ha='center', va='bottom', fontsize=12, rotation=30)
			for j in range(0,len(cell_types)):
				height = rects2[j].get_height()
				rect = rects2_2[j]
				ax2.text(rect.get_x() + rect.get_width()/2.0, 10, '%.1f' % height, ha='center', va='bottom', fontsize=12, rotation=30)
			plt.savefig('sensitivity_' + plt_name)
			plt.clf()
