from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import fnmatch
import numpy
import operator
import os
import pandas
import pickle
import sklearn
import sys
import utils


def train_main(classification_type, balanced, preprocessing, threshold, names_classifiers):
	train_data_dir = '/Cell_Profiler_Data/Stained_CP_Features'
	
	if classification_type == 'all':
		cell_types = ['lymphocyte', 'eosinophil', 'monocyte', 'neutrophil']
	elif classification_type == 'lymphocytes':
		cell_types = ['B', 'T']
		
	if preprocessing == '':
		preprocessing_ = ''
	else:
		preprocessing_ = '_' + preprocessing
	if preprocessing == 'low_var':
		output_dir = '/Classifiers/' + classification_type.capitalize() + '_' + balanced + preprocessing_ + '_' + str(threshold)
	else:
		output_dir = '/Classifiers/' + classification_type.capitalize() + '_' + balanced + preprocessing_
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		
	#Get the intersection of the zero score features over all subjects
	if preprocessing == 'remove_zero_features':
		if classification_type == 'all':
			zero_features_list_path = '/Machine_Learning_Outputs/White_Blood_Cells/Cross_Validation/All_' + balanced
		elif classification_type == 'lymphocytes':
			zero_features_list_path = '/Machine_Learning_Outputs/White_Blood_Cells/Cross_Validation/Lymphocytes_' + balanced
		files = fnmatch.filter(os.listdir(zero_features_list_path), 'GradientBoosting_feature_importances_*.txt')
		zero_features_list = []
		os.chdir(zero_features_list_path)
		for filename in files:
			lines = [line.rstrip('\n') for line in open(filename)]
			del lines[0]
			l = [e.split(":") for e in lines]
			dic = [[e[0], float(e[1])] for e in l]
			dic = dict(dic)
			zero_features_list.append([k for k,v in dic.items() if v==0])
		zero_features = set(zero_features_list[0]).intersection(*zero_features_list[:1])
		utils.write_zero_features_to_file(zero_features, output_dir)

	#Get the intersection of the top 10 features over all subjects
	if preprocessing == 'top_10_features':
		if classification_type == 'all':
			top_10_features_path = '/Machine_Learning_Outputs/White_Blood_Cells/Cross_Validation/All_' + balanced
		elif classification_type == 'lymphocytes':
			top_10_features_path = '/Machine_Learning_Outputs/White_Blood_Cells/Cross_Validation/Lymphocytes_' + balanced
		files = fnmatch.filter(os.listdir(top_10_features_path), 'GradientBoosting_feature_importances_*.txt')
		best_dicts_list = []
		os.chdir(top_10_features_path)
		for filename in files:
			lines = [line.rstrip('\n') for line in open(filename)]
			del lines[0]
			l = [e.split(":") for e in lines]
			dic = [[e[0], float(e[1])] for e in l]
			dic = dict(dic)
			best_dicts_list.append({k: v for k,v in dic.items() if v>=0.01})
		best_features_list = [set(dic.keys()) for dic in best_dicts_list]
		best_features_set = set.intersection(*best_features_list)
		best_scores = []
		best_dict = dict()
		for feature in best_features_set:
			best_scores = [dic.get(feature) for dic in best_dicts_list]
			best_dict.update({feature: numpy.mean(numpy.asarray(best_scores))})
		sorted_dict = sorted(best_dict.items(), key=operator.itemgetter(1))
		sorted_dict.reverse()
		sorted_dict = dict(sorted_dict)
		top_10_features = list(sorted_dict.keys())
		utils.write_top_10_features_to_file(sorted_dict, output_dir)

	#Features to exclude
	exclude_featuresDF = ['Location_CenterMassIntensity_X_DF_image', 'Location_CenterMassIntensity_Y_DF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_DF_image', 'Location_MaxIntensity_Y_DF_image', 'Number_Object_Number']
	exclude_featuresBF = ['ImageNumber', 'ObjectNumber', 'AreaShape_Center_X', 'AreaShape_Center_Y', 'AreaShape_EulerNumber', 'AreaShape_Orientation', 'AreaShape_Solidity', 'Location_CenterMassIntensity_X_BF_image', 'Location_CenterMassIntensity_Y_BF_image', 'Location_Center_X', 'Location_Center_Y', 'Location_MaxIntensity_X_BF_image', 'Location_MaxIntensity_Y_BF_image', 'Number_Object_Number']
	exclude_featuresDF.extend(exclude_featuresBF)
	exclude_features = exclude_featuresDF

	subject_ids = next(os.walk(train_data_dir))[1]

	data_filename = 'BF_cells_on_grid.txt';


	num_cells = []  #num_cells =[p1_num_cells, p2_num_cells, ..]: pi_num_cells = [#label1, #label2, ..]
	data = []
	for subject_id in subject_ids:
		subject_data = []
		num_cells_subject = []
		for cell_type in cell_types:
			if cell_type == 'lymphocyte':
				subject_cell_type_data = []
				for sub_label in ['B', 'T']:
					subject_sub_label_data = pandas.read_csv("".join([train_data_dir, "/", subject_id, "/", sub_label, "/", data_filename]), sep='\t')
					subject_cell_type_data.append(subject_sub_label_data)
				subject_cell_type_data = pandas.concat(subject_cell_type_data)
			else:
				subject_cell_type_data = pandas.read_csv("".join([train_data_dir, "/", subject_id, "/", cell_type, "/", data_filename]), sep='\t')
			ground_truth_list = [cell_type] * len(subject_cell_type_data)
			subject_cell_type_data = subject_cell_type_data.drop(exclude_features, axis=1)
			subject_cell_type_data = subject_cell_type_data[subject_cell_type_data.columns.drop(list(subject_cell_type_data.filter(regex='Marker_image')))]
			if preprocessing == 'remove_zero_features':
				subject_cell_type_data = subject_cell_type_data[subject_cell_type_data.columns.drop(zero_features)]
			if preprocessing == 'top_10_features':
				subject_cell_type_data = subject_cell_type_data[top_10_features]
			subject_cell_type_data = subject_cell_type_data.assign(ground_truth = ground_truth_list)
			subject_cell_type_data = subject_cell_type_data.dropna()
			subject_data.append(subject_cell_type_data)
			num_cells_subject.append(subject_cell_type_data.shape[0])
		if balanced == 'unbalanced':
			subject_data = pandas.concat(subject_data)
			subject_data = subject_data.dropna()
		data.append(subject_data)
		num_cells.append(num_cells_subject)
	
	#Get equal nums of cells from each cell type (balanced classes)
	if balanced == 'balanced':
		for j in range(0, len(data)):
			min_num_cells = min(num_cells[j])
			for i in range(0, len(cell_types)):
				data[j][i] = data[j][i].sample(min_num_cells)
			data[j] = pandas.concat(data[j])
	utils.write_num_cells_to_file(num_cells, subject_ids, cell_types, output_dir)
	
	#Merge train dataframes
	train_data= pandas.concat(data)
	train_data = train_data.dropna()
			
	#Separate ground truth from the rest
	ground_truth = train_data['ground_truth']
	train_data = train_data.drop('ground_truth', axis = 1)

	#Features standardization using z-score normalization
	if preprocessing == 'norm':
		norm_train_data = pandas.DataFrame(columns=train_data.columns)
		for feature_name in train_data.columns:
			norm_train_data[feature_name] = (train_data[feature_name] - train_data[feature_name].mean()) / train_data[feature_name].std()
		train_data = norm_train_data
	#Remove low variance features
	if preprocessing == 'low_var':
		selector = VarianceThreshold(threshold)
		columns = train_data.columns
		train_data = selector.fit_transform(train_data)
		labels = [columns[i] for i in selector.get_support(indices=True)]
		train_data = pandas.DataFrame(train_data, columns=labels)
	
	
	for name, classifier in names_classifiers:
		classifier.fit(train_data, ground_truth)
		if hasattr(classifier, 'feature_importances_'):
			feature_names = train_data.columns
			feature_importances = sorted(zip(classifier.feature_importances_, feature_names),reverse=True)
			utils.write_feature_importances_to_file(feature_importances, name, output_dir)
		#save classifier to file
		pkl_filename = name + "_classifier.pkl"
		with open(pkl_filename, 'wb') as file:
			pickle.dump(classifier, file)
	
