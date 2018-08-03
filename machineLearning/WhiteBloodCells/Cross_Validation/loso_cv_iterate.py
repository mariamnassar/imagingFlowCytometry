from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import fnmatch
import numpy
import operator
import os
import pandas
import sklearn
import sys
import utils
from loso_cv_main import loso_cv_main


names_classifiers = []
names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))
names_classifiers.append(('KNN', KNeighborsClassifier()))
names_classifiers.append(('SVC', SVC()))
names_classifiers.append(('AdaBoost', AdaBoostClassifier()))
	
preprocessing_types = ['', 'low_var', 'norm', 'remove_zero_features', 'top_10_features']
thresholds = [0.001, 0.0001, 0.00001]
for classification_type in ['lymphocytes', 'all']:
	for balanced in ['balanced', 'unbalanced']:
		for preprocessing in preprocessing_types:
			if preprocessing == 'low_var':
				for threshold in thresholds:
					loso_cv_main(classification_type, balanced, preprocessing, threshold, names_classifiers)
			else:
				loso_cv_main(classification_type, balanced, preprocessing, None)
