import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from classification_HL import *
from scipy.signal import find_peaks
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Feature selection using information gain
def select_by_information_gain(X, y, feature_num):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    top_indices = np.argsort(feature_importances)[-feature_num:]
    return top_indices

# Feature selection using mutual information
def select_by_mutual_information(X, y, feature_num):
    mutual_info = mutual_info_classif(X, y)
    top_indices = np.argsort(mutual_info)[-feature_num:]
    return top_indices

# Feature selection using K-Means clustering
def select_by_kmeans(X, feature_num):
    kmeans = KMeans(n_clusters=feature_num, random_state=42)
    kmeans.fit(X.T)  # Cluster on features (transpose)
    centers = kmeans.cluster_centers_
    cluster_indices = []
    for center in centers:
        closest_idx = np.argmin(np.linalg.norm(X.T - center, axis=1))
        cluster_indices.append(closest_idx)
    return cluster_indices

# Feature selection using diversity sampling
def select_by_diversity_sampling(X, feature_num):
    selected_indices = []
    remaining_indices = list(range(X.shape[1]))
    selected_indices.append(remaining_indices.pop(0))  # Start with the first feature

    while len(selected_indices) < feature_num:
        max_distance = 0
        next_index = -1
        for i in remaining_indices:
            distances = pairwise_distances(X[:, selected_indices].T, X[:, i].reshape(1, -1))
            min_distance = np.min(distances)
            if min_distance > max_distance:
                max_distance = min_distance
                next_index = i
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    return selected_indices

# Load data
file_path = r"J:\paper 2 analyse\ASD dataset\train_data_average.xlsx"
file_path_test = r"J:\paper 2 analyse\ASD dataset\test_data_average.xlsx"
df = pd.read_excel(file_path, sheet_name='snv')
save_dir = r'J:\paper 2 analyse\ASD result\no resample ASD feature selection snv'
df_test = pd.read_excel(file_path_test, sheet_name='snv')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

feature_num_list = [3, 5, 10]
wave = df.columns[4:-2].to_numpy()

# Data preprocessing
X_train = df.iloc[:, 4:-2].values
y_train = df.iloc[:, -1].values
X_test = df_test.iloc[:, 4:-2].values
y_test = df_test.iloc[:, -1].values

# SVM parameters
svm_kernel = 'linear'
svm_kernel_YX = ['linear', 'rbf', 'poly']
svm_coef0_range = [0, 50]
svm_c_range = [1, 100]
svm_gamma_range = [0.01, 50]
svm_tol = 1e-3

opt_num_iter_cv = 5
opt_num_fold_cv = 5
opt_num_evals = 3

# Train full model
best_model = tune_svm_classification(X_train, y_train,
                                     svm_kernel=svm_kernel,
                                     svm_c_range=svm_c_range,
                                     svm_gamma_range=svm_gamma_range,
                                     svm_tol=svm_tol,
                                     opt_num_iter_cv=opt_num_iter_cv,
                                     opt_num_fold_cv=opt_num_fold_cv,
                                     opt_num_evals=opt_num_evals)

y_pred = best_model.predict(X_test)
full_data_f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score with full features: ", full_data_f1)
