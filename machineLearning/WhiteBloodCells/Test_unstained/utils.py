import matplotlib
import matplotlib.pyplot as plt
import numpy
import os


def write_wbc_count_to_file(wbc_count, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.chdir(output_dir)
	wbc_count_file_name = 'wbc_count.txt'
	numpy.savetxt(''.join(output_dir + '/' + wbc_count_file_name), wbc_count, fmt='%.3f')

	
def plot_wbc_count(wbc_count_patient, wbc_counts_ref, comp_to, cell_types, classifier, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.chdir(output_dir)
	bar_labels = cell_types
	bar_width = 0.3
	opacity = 0.8
	y_pos = numpy.arange(len(bar_labels))
	plt.bar(y_pos, wbc_count_patient, bar_width, alpha=opacity, align='center', color='red', label='Patient')
	plt.bar(y_pos + bar_width, wbc_counts_ref, bar_width, alpha=opacity, align='center', color='green', label=comp_to.capitalize())
	plt.ylabel('Percent of cells')
	plt.xticks(y_pos + bar_width/2, bar_labels)
	plt.ylim(0.0, 100)
	plot_file_name = classifier + '_wbc_count_' + comp_to + '.png'
	plt.legend()
	plt.savefig(plot_file_name)
	plt.clf()
	
	
def plot_wbc_count_avg_comp(wbc_count_average, num_cells_list, wbc_counts_ref, cell_types, classifier, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.chdir(output_dir)
	bar_labels = cell_types
	bar_width = 0.3
	opacity = 0.8
	y_pos = numpy.arange(len(bar_labels))
	plt.bar(y_pos, wbc_count_average, bar_width, alpha=opacity, align='center', color='red', label='Average WBC count ML')
	plt.bar(y_pos + bar_width, num_cells_list, bar_width, alpha=opacity, align='center', color='blue', label='Average WBC count Gating')
	plt.bar(y_pos + 2*bar_width, wbc_counts_ref, bar_width, alpha=opacity, align='center', color='green', label='Reference WBC count')
	plt.ylabel('Percent of cells')
	plt.xticks(y_pos + bar_width, bar_labels)
	plt.ylim(0.0, 100)
	plot_file_name = classifier + '_wbc_counts_comp' + '.png'
	plt.legend()
	plt.savefig(plot_file_name)
	plt.clf()
