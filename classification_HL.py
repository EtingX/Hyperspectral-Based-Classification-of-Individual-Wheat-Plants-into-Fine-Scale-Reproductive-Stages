########################################################################################################################
import os
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

# 屏蔽特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
def plot_samples_with_colourbar(samples, labels, wavelengths=[], input_type='Input data values', title='Title'):
    """
    Plot the samples with a colour bar which is defined by the values of the labels.

    :param samples: input data array; usually reflectance values
    :param labels: the values of the labels. E.g. 0, 1, 2 ......
    :param x_axis_value: the x_axis_value in the plot; default is [] which will lead to x axis values of 1, 2, 3 ......
    :param input_type: A string of "reflectance", "pca", etc
    :param title: title for plot
    :return: return 0 if no errors

    Version 1.0 Date: Aug 25, 2021
    Author: Huajian Liu huajian.liu@adelaide.edu.au
    """

    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import numpy as np

    uni_label = np.unique(labels)
    cmap = plt.get_cmap('jet', len(uni_label))
    norm = mpl.colors.BoundaryNorm(np.arange(len(uni_label) + 1)-0.5, len(uni_label))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if wavelengths == []:
        plt.figure()
        plt.title(title)
        for i in range(samples.shape[0]):
            ind_color = np.where(uni_label == labels[i])[0][0]
            plt.plot(samples[i], c=cmap(ind_color))
        plt.xlabel('Dimensions of ' + input_type, fontsize=12, fontweight='bold')
    else:
        plt.figure()
        plt.title(title)
        for i in range(samples.shape[0]):
            ind_color = np.where(uni_label == labels[i])[0][0]
            plt.plot(wavelengths, samples[i], c=cmap(ind_color))
        plt.xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')

    plt.colorbar(sm, ticks=uni_label)
    plt.ylabel(input_type, fontsize=12, fontweight='bold')


def calculate_bi_classification_metrics(label, output):
    """
    Calculate the metrics of binary classification.

    :param label:
    :param output:
    :return: A dictionary of the metrics
    """
    import sklearn.metrics as met

    # Confusion matrix
    conf_max = met.confusion_matrix(label, output)

    # Recall
    recall = met.recall_score(label, output)

    # Precision
    precision = met.precision_score(label, output)

    # f1 score
    f1 = met.f1_score(label, output)

    # Accuracy
    accuracy = met.accuracy_score(label, output)

    dict = {'confusion_mat': conf_max,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1}
    return dict

def average_bi_classification_metrics(record_cv):
    import numpy as np

    # For recording average
    ave_con_mat_tra = []
    ave_precision_tra = []
    ave_recall_tra = []
    ave_accuracy_tra = []
    ave_f1_tra = []

    ave_con_mat_val = []
    ave_precision_val = []
    ave_recall_val = []
    ave_accuracy_val = []
    ave_f1_val = []

    for a_record in record_cv:
        # Prepare data for averaging
        ave_con_mat_tra.append(a_record['classification_metrics_tra']['confusion_mat'])
        ave_accuracy_tra.append(a_record['classification_metrics_tra']['accuracy'])
        ave_recall_tra.append(a_record['classification_metrics_tra']['recall'])
        ave_precision_tra.append(a_record['classification_metrics_tra']['precision'])
        ave_f1_tra.append(a_record['classification_metrics_tra']['f1'])

        ave_con_mat_val.append(a_record['classification_metrics_val']['confusion_mat'])
        ave_accuracy_val.append(a_record['classification_metrics_val']['accuracy'])
        ave_recall_val.append(a_record['classification_metrics_val']['recall'])
        ave_precision_val.append(a_record['classification_metrics_val']['precision'])
        ave_f1_val.append(a_record['classification_metrics_val']['f1'])

    # Average confusion matrix
    ave_con_mat_tra = np.asarray(ave_con_mat_tra)
    ave_con_mat_tra = np.mean(ave_con_mat_tra, axis=0)
    ave_con_mat_tra = np.round(ave_con_mat_tra).astype(np.int)

    ave_con_mat_val = np.asarray(ave_con_mat_val)
    ave_con_mat_val = np.mean(ave_con_mat_val, axis=0)
    ave_con_mat_val = np.round(ave_con_mat_val).astype(np.int)

    # Average accuracy
    ave_accuracy_tra = np.asarray(ave_accuracy_tra)
    ave_accuracy_tra = np.mean(ave_accuracy_tra, axis=0)
    ave_accuracy_tra = np.round(ave_accuracy_tra, 4)

    ave_accuracy_val = np.asarray(ave_accuracy_val)
    ave_accuracy_val = np.mean(ave_accuracy_val, axis=0)
    ave_accuracy_val = np.round(ave_accuracy_val, 4)

    # Average recall
    ave_recall_tra = np.asarray(ave_recall_tra)
    ave_recall_tra = np.mean(ave_recall_tra, axis=0)
    ave_recall_tra = np.round(ave_recall_tra, 4)

    ave_recall_val = np.asarray(ave_recall_val)
    ave_recall_val = np.mean(ave_recall_val, axis=0)
    ave_recall_val = np.round(ave_recall_val, 4)

    # Average precision
    ave_precision_tra = np.asarray(ave_precision_tra)
    ave_precision_tra = np.mean(ave_precision_tra, axis=0)
    ave_precision_tra = np.round(ave_precision_tra, 4)

    ave_precision_val = np.asarray(ave_precision_val)
    ave_precision_val = np.mean(ave_precision_val, axis=0)
    ave_precision_val = np.round(ave_precision_val, 4)

    # Average f1 score
    ave_f1_tra = np.asarray(ave_f1_tra)
    ave_f1_tra = np.mean(ave_f1_tra, axis=0)
    ave_f1_tra = np.round(ave_f1_tra, 2)

    ave_f1_val = np.asarray(ave_f1_val)
    ave_f1_val = np.mean(ave_f1_val, axis=0)
    ave_f1_val = np.round(ave_f1_val, 2)

    ave_metrics = {'ave_metrics_train': {'conf_mat': str(ave_con_mat_tra),
                                         'accuracy': ave_accuracy_tra,
                                         'recall': ave_recall_tra,
                                         'precision': ave_precision_tra,
                                         'f1': ave_f1_tra},
                   'ave_metrics_validation': {'conf_mat': str(ave_con_mat_val),
                                              'accuracy': ave_accuracy_val,
                                              'recall': ave_recall_val,
                                              'precision': ave_precision_val,
                                              'f1': ave_f1_val}}

    return ave_metrics


def average_classification_metrics(record_cv):
    """
    Average classification metrics for multiple classes. Designed for repeated_k-fold_cv.
    :param record_cv: recored of cross validaton retruned from
    :return: Averaged classification metrics.
    """
    import numpy as np

    # For recording average
    ave_con_mat_tra = []
    ave_precision_tra = []
    ave_recall_tra = []
    ave_accuracy_tra = []
    ave_f1_tra = []

    ave_con_mat_val = []
    ave_precision_val = []
    ave_recall_val = []
    ave_accuracy_val = []
    ave_f1_val = []

    for a_record in record_cv:
        # Prepare data for averaging
        ave_con_mat_tra.append(a_record['Confusion matrix of train'])
        ave_accuracy_tra.append(a_record['Classification report of train']['accuracy'])
        ave_recall_tra.append(a_record['Classification report of train']['weighted avg']['recall'])
        ave_precision_tra.append(a_record['Classification report of train']['weighted avg']['precision'])
        ave_f1_tra.append(a_record['Classification report of train']['weighted avg']['f1-score'])

        ave_con_mat_val.append(a_record['Confusion matrix of validation'])
        ave_accuracy_val.append(a_record['Classification report of validation']['accuracy'])
        ave_recall_val.append(a_record['Classification report of validation']['weighted avg']['recall'])
        ave_precision_val.append(a_record['Classification report of validation']['weighted avg']['precision'])
        ave_f1_val.append(a_record['Classification report of validation']['weighted avg']['f1-score'])

    # Average confusion matrix
    ave_con_mat_tra = np.asarray(ave_con_mat_tra)
    ave_con_mat_tra = np.mean(ave_con_mat_tra, axis=0)
    ave_con_mat_tra = np.round(ave_con_mat_tra).astype(np.int32)

    ave_con_mat_val = np.asarray(ave_con_mat_val)
    ave_con_mat_val = np.mean(ave_con_mat_val, axis=0)
    ave_con_mat_val = np.round(ave_con_mat_val).astype(np.int32)

    # Average accuracy
    ave_accuracy_tra = np.asarray(ave_accuracy_tra)
    ave_accuracy_tra = np.mean(ave_accuracy_tra, axis=0)
    ave_accuracy_tra = np.round(ave_accuracy_tra, 4)

    ave_accuracy_val = np.asarray(ave_accuracy_val)
    ave_accuracy_val = np.mean(ave_accuracy_val, axis=0)
    ave_accuracy_val = np.round(ave_accuracy_val, 4)

    # Average recall
    ave_recall_tra = np.asarray(ave_recall_tra)
    ave_recall_tra = np.mean(ave_recall_tra, axis=0)
    ave_recall_tra = np.round(ave_recall_tra, 4)

    ave_recall_val = np.asarray(ave_recall_val)
    ave_recall_val = np.mean(ave_recall_val, axis=0)
    ave_recall_val = np.round(ave_recall_val, 4)

    # Average precision
    ave_precision_tra = np.asarray(ave_precision_tra)
    ave_precision_tra = np.mean(ave_precision_tra, axis=0)
    ave_precision_tra = np.round(ave_precision_tra, 4)

    ave_precision_val = np.asarray(ave_precision_val)
    ave_precision_val = np.mean(ave_precision_val, axis=0)
    ave_precision_val = np.round(ave_precision_val, 4)

    # Average f1 score
    ave_f1_tra = np.asarray(ave_f1_tra)
    ave_f1_tra = np.mean(ave_f1_tra, axis=0)
    ave_f1_tra = np.round(ave_f1_tra, 2)

    ave_f1_val = np.asarray(ave_f1_val)
    ave_f1_val = np.mean(ave_f1_val, axis=0)
    ave_f1_val = np.round(ave_f1_val, 2)

    ave_metrics = {'ave_metrics_train': {'conf_mat': str(ave_con_mat_tra),
                                         'accuracy': ave_accuracy_tra,
                                         'recall': ave_recall_tra,
                                         'precision': ave_precision_tra,
                                         'f1': ave_f1_tra},
                   'ave_metrics_validation': {'conf_mat': str(ave_con_mat_val),
                                              'accuracy': ave_accuracy_val,
                                              'recall': ave_recall_val,
                                              'precision': ave_precision_val,
                                              'f1': ave_f1_val}}

    return ave_metrics



def repeadted_kfold_cv(input, label, key,model_save_dir,n_splits, n_repeats, tune_model, karg, random_state=0,
                       note='',
                       flag_save=False,xgboost=False):
    """
    Perform repeated k-folds cross validation of classification.

    :param input: Input data in the format of 2D numpy array.
    :param label: The ground-trued labels. 1D numpy array in int.
    :param n_splits: The number of splits for cross validation
    :param n_repeats: The number of repeat for cross validation.
    :param tune_model: The function for tuning the models.
    :param karg: Key words arguments for tune_model()
    :param random_state: Random state for cross-validation. Default is 0
    :param flag_save: Flag to save the record. If set to True, it will save the record as a .save file in the present
           working directory. Default is False
    :param file_name_save: The file name to save the record. Default is 'cv_record'.
    :return: If have valid record, it returns a dictionary recording the report of repeated cross validation; otherwise
           it return -1.

     Version 1.0 Date: Aug 25, 2021 Tested for binary classification.
     Author: Huajian Liu huajian.liu@adelaide.edu.au
    """

    from sklearn.model_selection import RepeatedKFold
    import numpy as np
    import joblib
    from datetime import datetime
    import sklearn.metrics as met

    # Performing repeated cross validation
    # First, calculate the number of samples of each class; total samples
    classes, counts = np.unique(label, return_counts=True)
    total_samples = label.shape[0]

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    record_each_cv = []
    count_cv = 1
    for train_ind, val_index in rkf.split(input):
        print('')
        print('========== cross validation ' + str(count_cv) + '==========')

        # Train
        tuned_model = tune_model(input[train_ind], label[train_ind], **karg)
        print('Tuned model:')
        print(tuned_model)
        save_tuned_model = tuned_model
        # Number of positive and negative samples in validation
        classes_val, counts_val = np.unique(label[val_index], return_counts=True)
        classes_tra, counts_tra = np.unique(label[train_ind], return_counts=True)

        # The classes of training or validation should be the same of the total classes
        if classes_tra.shape[0] != classes.shape[0]:
            print('The classes of training is ', classes_tra)
            print('Do not take into account in cv.')
        elif classes_val.shape[0] != classes.shape[0]:
            print("The classes of validation is ", classes_val)
            print('Do not take into account in cv.')
        else:
            if xgboost==True:
                print('XGBOOST!')
                # Convert input data to DMatrix for prediction
                dtrain_input = xgb.DMatrix(input[train_ind], label=label[train_ind])
                dval_input = xgb.DMatrix(input[val_index], label=label[val_index])

                # Prediction using XGBoost
                output_tra = tuned_model.predict(dtrain_input)
                output_val = tuned_model.predict(dval_input)

                # Classification report for binary or multiple classes.
                report_tra = met.classification_report(label[train_ind], output_tra, output_dict=True)
                report_val = met.classification_report(label[val_index], output_val, output_dict=True)

                # Confusion matrix

            elif xgboost==False:
                # Prediction
                output_tra = tuned_model.predict(input[train_ind])
                output_val = tuned_model.predict(input[val_index])

                # Classification report for binary of multiple classes.
                report_tra = met.classification_report(label[train_ind], output_tra, output_dict=True)
                report_val = met.classification_report(label[val_index], output_val, output_dict=True)

            # Confusion matrix
            conf_max_tra = met.confusion_matrix(label[train_ind], output_tra)
            conf_max_val = met.confusion_matrix(label[val_index], output_val)

            print('-----------------')
            print('Train')
            print('-----------------')
            print('Classes', classes_tra)
            print('Count of each class:')
            print(counts_tra)
            print(report_tra)

            print('-----------------')
            print('Validation')
            print('-----------------')
            print('Classes', classes_val)
            print('Count of each class:')
            print(counts_val)
            print(report_val)

            # Record
            record_each_cv.append({'Classification report of validation': report_val,
                                   'Count of validation': counts_val,
                                   'Classification report of train': report_tra,
                                   'Count of train': counts_val,
                                   'Confusion matrix of train': conf_max_tra,
                                   'Confusion matrix of validation': conf_max_val,
                                   'Classes': classes_tra})
            count_cv += 1

    if record_each_cv.__len__() == 0:
        print('No valid record in cross-validation.')
        return -1

    # Average
    ave_metrics = average_classification_metrics(record_each_cv)

    # Print the summary of cross-validation
    print()
    print('Summary of ' + str(n_splits) + '-fold cross validation with ' + str(n_repeats) + ' repeats')
    print('Total samples: ', total_samples)
    print('Classes:', classes)
    print('Count in each classes: ', counts)
    print('Average metrics of train: ')
    print(ave_metrics['ave_metrics_train'])
    print('Average metrics of validation: ')
    print(ave_metrics['ave_metrics_validation'])

    # Train a final model using all of the data
    final_model = tune_model(input, label, **karg)

    # Record
    record = {'record of each cv': record_each_cv,
# <<<<<<< HEAD
#               'average confusion matrix': ave_con_mat,
#               'average recall': ave_recall,
#               'average f1': ave_f1,
#               'average accuracy': ave_asccuracy,
#               'average precision': ave_precision,
# =======
              'average metrics': ave_metrics,
# >>>>>>> c2a56474bae101bc14180e6b2aa619d5aeda3d21
              'total samples': total_samples,
              'classes': str(classes),
              'count in each class': str(counts),
              'final model': final_model,
              'Note': note}

    # Save
    if flag_save:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        file_name_save = model_save_dir + '/' + datetime.now().strftime('%y-%m-%d-%H-%M-%S') + str(key)+ '.sav'
        joblib.dump(record, file_name_save)
        return record, file_name_save, save_tuned_model
    else:
        return record, save_tuned_model


def tune_svm_classification(input,
                            label,
                            svm_kernel='rbf',
                            svm_c_range=[1, 100],
                            svm_gamma_range=[1, 50],
                            svm_tol=1e-3,
                            opt_num_iter_cv=5,
                            opt_num_fold_cv=5,
                            opt_num_evals=5):
    """
    Tune a support vector machine classificaition model based on sklearn.svm.SVC
    :param input: The input data for training the model. 2D numpy array
    :param label: The ground-trued labels for training the model. 1D numpy array in int.
    :param svm_kernel: The kernel function of SVM. Refer to sklearn.svm.SVC
    :param svm_c_range: The searching range of C of sklearn.svm.SVC. Default is [1, 100]
    :param svm_gamma_range: The searching range of gamma of sklearn.svm.SVC. Defaul is [1, 50]
    :param svm_tol: The tol value of sklearn.svm.SVC.
    :param opt_num_iter_cv: The number of iteration of cross validation of optunity.
    :param opt_num_fold_cv: The number of fold of cross validation of optunity.
    :param opt_num_evals: The number of evaluation of optunity.
    :return: A tuned SVM model for binary classification.

    Author: Huajina Liu email: huajina.liu@adelaide.edu.au
    Version: 1.0 Date: August 20, 2021
    """
    from sklearn.svm import SVC
    import optunity.metrics

    # Search the optimal parameters
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = SVC(kernel=svm_kernel, C=C, gamma=gamma, tol=svm_tol)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions) # error_rate = 1.0 - accuracy(y, yhat)

    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=opt_num_evals, C=svm_c_range, gamma=svm_gamma_range)

    # Train a model using the optimal parameters
    tuned_model = SVC(probability=True, **optimal_pars).fit(input, label)
    return tuned_model


def tune_random_forest_classification(input,
                                      label,
                                      rf_n_estimators_range=[50, 200],
                                      rf_max_depth_range=[5, 50],
                                      rf_min_samples_split_range=[2, 10],
                                      rf_min_samples_leaf_range=[1, 5],
                                      opt_num_iter_cv=5,
                                      opt_num_fold_cv=5,
                                      opt_num_evals=5):
    """
    Tune a Random Forest classification model based on sklearn.ensemble.RandomForestClassifier.
    :param input: The input data for training the model. 2D numpy array.
    :param label: The ground-truth labels for training the model. 1D numpy array in int.
    :param rf_n_estimators_range: The search range for the number of estimators (trees) in the Random Forest. Default is [50, 200].
    :param rf_max_depth_range: The search range for max depth of each tree in the Random Forest. Default is [5, 50].
    :param rf_min_samples_split_range: The search range for min samples required to split an internal node. Default is [2, 10].
    :param rf_min_samples_leaf_range: The search range for min samples required at a leaf node. Default is [1, 5].
    :param opt_num_iter_cv: The number of iterations for cross-validation in Optunity.
    :param opt_num_fold_cv: The number of folds for cross-validation in Optunity.
    :param opt_num_evals: The number of evaluations in Optunity.
    :return: A tuned Random Forest model for binary classification.
    """

    from sklearn.ensemble import RandomForestClassifier
    import optunity.metrics

    # Define the cross-validated tuning function
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       min_samples_leaf=int(min_samples_leaf),
                                       random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions)  # error_rate = 1.0 - accuracy

    # Perform hyperparameter tuning using Optunity
    optimal_pars, _, _ = optunity.minimize(tune_cv,
                                           num_evals=opt_num_evals,
                                           n_estimators=rf_n_estimators_range,
                                           max_depth=rf_max_depth_range,
                                           min_samples_split=rf_min_samples_split_range,
                                           min_samples_leaf=rf_min_samples_leaf_range)

    # Train the final model using optimal parameters
    tuned_model = RandomForestClassifier(n_estimators=int(optimal_pars['n_estimators']),
                                         max_depth=int(optimal_pars['max_depth']),
                                         min_samples_split=int(optimal_pars['min_samples_split']),
                                         min_samples_leaf=int(optimal_pars['min_samples_leaf']),
                                         random_state=42)
    tuned_model.fit(input, label)
    return tuned_model



def tune_svm_classification_YX(input,
                            label,
                            svm_kernel_options=['linear', 'rbf', 'poly'],
                            svm_c_range=[1, 100],
                            svm_gamma_range=[0.01, 1],
                            svm_coef0_range=[0, 10],
                            svm_tol=1e-3,
                            opt_num_iter_cv=5,
                            opt_num_fold_cv=5,
                            opt_num_evals=5):
    """
    Tune a support vector machine classification model based on sklearn.svm.SVC
    :param input: The input data for training the model. 2D numpy array
    :param label: The ground-truth labels for training the model. 1D numpy array in int.
    :param svm_kernel_options: List of kernel options to try (e.g., ['linear', 'rbf', 'poly', 'sigmoid']).
    :param svm_c_range: The search range for C in sklearn.svm.SVC. Default is [1, 100].
    :param svm_gamma_range: The search range for gamma in sklearn.svm.SVC. Default is [0.01, 1].
    :param svm_coef0_range: The search range for coef0 in sklearn.svm.SVC when using 'poly' or 'sigmoid' kernel. Default is [0, 10].
    :param svm_tol: The tolerance for stopping criterion of sklearn.svm.SVC.
    :param opt_num_iter_cv: The number of cross-validation iterations in optunity.
    :param opt_num_fold_cv: The number of cross-validation folds in optunity.
    :param opt_num_evals: The number of evaluations in optunity.
    :return: A tuned SVM model for binary classification.

    Author: Huajina Liu email: huajina.liu@adelaide.edu.au
    Version: 1.2 Date: Updated
    """
    from sklearn.svm import SVC
    import optunity.metrics

    # Define cross-validated tuning function
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma, coef0, kernel_index):
        # 根据 kernel_index 选择核函数
        kernel = svm_kernel_options[int(kernel_index)]

        # 初始化模型，适配各核函数
        model = SVC(
            kernel=kernel, C=C,
            gamma=gamma if kernel in ['rbf', 'poly'] else 'scale',
            coef0=coef0 if kernel == 'poly' else 0,  # Only use coef0 for 'poly' and 'sigmoid'
            tol=svm_tol
        )

        # 训练并预测
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions)  # error_rate = 1.0 - accuracy

    # 定义搜索空间，包括 coef0 和 kernel_index
    optimal_pars, _, _ = optunity.minimize(
        tune_cv,
        num_evals=opt_num_evals,
        C=svm_c_range,
        gamma=svm_gamma_range,
        coef0=svm_coef0_range,
        kernel_index=[0, len(svm_kernel_options) - 1]
    )

    # 映射最佳 kernel_index 参数到实际核函数
    optimal_kernel = svm_kernel_options[int(optimal_pars['kernel_index'])]

    # 使用最佳参数训练模型
    tuned_model = SVC(
        probability=True,
        kernel=optimal_kernel,
        C=optimal_pars['C'],
        gamma=optimal_pars['gamma'] if optimal_kernel in ['rbf', 'poly'] else 'scale',
        coef0=optimal_pars['coef0'] if optimal_kernel == 'poly' else 0,
        tol=svm_tol
    ).fit(input, label)

    return tuned_model


def tune_nn_classification(input,
                           label,
                           hidden_layer_sizes_range=[(10,), (50,), (100,), (100, 50), (100, 10), (50, 10), (100, 50,10)],
                           activation_options=['relu', 'tanh'],
                           alpha_range=[0.0001, 0.01],
                           learning_rate_init_range=[0.001, 0.1],
                           solver_options=['adam', 'sgd'],
                           opt_num_iter_cv=5,
                           opt_num_fold_cv=5,
                           opt_num_evals=5):
    """
    Tune a neural network (MLPClassifier) for binary or multiclass classification.
    :param input: The input data for training the model. 2D numpy array.
    :param label: The ground-truth labels for training the model. 1D numpy array in int.
    :param hidden_layer_sizes_range: List of possible hidden layer configurations.
    :param activation_options: List of activation functions to try.
    :param alpha_range: Range for L2 regularization parameter (alpha).
    :param learning_rate_init_range: Range for initial learning rate.
    :param solver_options: List of optimization solvers to try.
    :param opt_num_iter_cv: Number of cross-validation iterations for optunity.
    :param opt_num_fold_cv: Number of folds in cross-validation for optunity.
    :param opt_num_evals: Number of evaluations for optunity optimization.
    :return: A tuned MLPClassifier model.
    """

    from sklearn.neural_network import MLPClassifier
    import optunity.metrics
    import optunity

    # Cross-validated tuning function
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, hidden_layer_index, activation_index, alpha, learning_rate_init, solver_index):
        # Convert indices to actual values
        hidden_layer_sizes = hidden_layer_sizes_range[int(hidden_layer_index)]
        activation = activation_options[int(activation_index)]
        solver = solver_options[int(solver_index)]

        # Initialize and train the model
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              activation=activation,
                              alpha=alpha,
                              learning_rate_init=learning_rate_init,
                              solver=solver,
                              max_iter=500,
                              random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        # Return error rate
        return optunity.metrics.error_rate(y_test, predictions)

    # Define the search space
    search_space = {
        'hidden_layer_index': [0, len(hidden_layer_sizes_range) - 1],
        'activation_index': [0, len(activation_options) - 1],
        'alpha': alpha_range,
        'learning_rate_init': learning_rate_init_range,
        'solver_index': [0, len(solver_options) - 1]
    }

    # Optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=opt_num_evals, **search_space)

    # Convert optimal parameters to actual values
    best_hidden_layer_sizes = hidden_layer_sizes_range[int(optimal_pars['hidden_layer_index'])]
    best_activation = activation_options[int(optimal_pars['activation_index'])]
    best_solver = solver_options[int(optimal_pars['solver_index'])]
    best_alpha = optimal_pars['alpha']
    best_learning_rate_init = optimal_pars['learning_rate_init']

    # Train final model with optimal parameters
    tuned_model = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes,
                                activation=best_activation,
                                alpha=best_alpha,
                                learning_rate_init=best_learning_rate_init,
                                solver=best_solver,
                                max_iter=300,
                                random_state=42).fit(input, label)

    return tuned_model


def tune_xgboost_YX(X_train, y_train, opt_num_iter_cv=5, opt_num_fold_cv=5, opt_num_evals=10):
    """
    使用 Optuna 微调 XGBoost 模型，支持 GPU 和 CPU。
    """
    # 检测 GPU 可用性
    gpu_available = False
    try:
        dmatrix = xgb.DMatrix(X_train[:10], label=y_train[:10])
        params_test = {'tree_method': 'gpu_hist', 'objective': 'reg:squarederror'}
        xgb.train(params_test, dmatrix, num_boost_round=1)
        gpu_available = True
    except Exception as e:
        print(f"GPU test failed: {e}")
        gpu_available = False

    tree_method = 'hist' if not gpu_available else 'hist'
    device = 'cuda' if gpu_available else 'cpu'
    print(f"{'GPU detected' if gpu_available else 'No GPU detected'}, using {device} for training.")

    def objective(trial):
        # 超参数搜索空间
        param_grid = {
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'mlogloss',
            'tree_method': tree_method,
            'device': device,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }

        skf = StratifiedKFold(n_splits=opt_num_fold_cv, shuffle=True, random_state=42)
        accuracy_scores = []

        for train_idx, valid_idx in skf.split(X_train, y_train):
            # 分割数据集
            X_tr, X_val = X_train[train_idx], X_train[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]

            # 转换为 DMatrix 格式
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_val, label=y_val)

            # 使用 xgb.train 和 Optuna 的 Pruning 回调
            pruning_callback = XGBoostPruningCallback(trial, "validation-mlogloss")
            bst = xgb.train(
                param_grid,
                dtrain,
                num_boost_round=300,
                evals=[(dvalid, "validation")],
                early_stopping_rounds=10,
                callbacks=[pruning_callback],
                verbose_eval=False,
            )

            # 使用验证集评估
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(y_val, pred_labels)
            accuracy_scores.append(accuracy)

        return -1 * np.mean(accuracy_scores)

    # 记录最佳结果
    best_trial_overall = None
    for iter_num in range(opt_num_iter_cv):
        print(f"Starting Experiment {iter_num + 1}/{opt_num_iter_cv}...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=opt_num_evals)

        if best_trial_overall is None or study.best_trial.value < best_trial_overall.value:
            best_trial_overall = study.best_trial

    print("Best Trial Overall:")
    print("  Value (Negative Accuracy):", best_trial_overall.value)
    print("  Params:", best_trial_overall.params)

    # 使用最佳参数重新训练模型
    final_params = {**best_trial_overall.params,
                    'objective': 'multi:softmax',
                    'num_class': len(np.unique(y_train)),
                    'eval_metric': 'mlogloss',
                    'tree_method': tree_method,
                    'device': device}

    final_dtrain = xgb.DMatrix(X_train, label=y_train)
    final_model = xgb.train(final_params, final_dtrain, num_boost_round=300)

    return final_model

def make_class_map(hyp_data,
                   wavelength,
                   seg_model_path,
                   seg_model_name,
                   classification_model_path,
                   classification_model_name,
                   band_r=97,
                   band_g=52,
                   band_b=14,
                   gamma=0.7,
                   flag_remove_noise=True,
                   flag_figure=False):

    """
    Make a 2D classification map of a 3D hypercube based on trained crop segmentation model and classification model.

    :param hyp_data: A calibrated 3D hypercube of row x col x dim. Float format in the range of [0, 1]
    :param wavelength: The corresponding wavelengths of the hypercube.
    :param seg_model_path: The path to save the crop segmentation model.
    :param seg_model_name: The name of the crop segmentation model.
    :param classification_model_path: The path of the classification model.
    :param classification_model_name: The name of the classification model.
    :param band_r: The red band number to create a pseudo RGB image. Default is 97.
    :param band_g: The green band number to create a pseudo RGB image. Default is 52.
    :param band_b: The blue band number to create a pseddo RGB image. Default is 14.
    :param gamma: The gamma value for exposure adjustment. Default is 0.7
    :param flag_remove_noise: The flat to remove noise or not in the crop-segmented image.
    :param flag_figure: The flat to show the result or not.
    :return: A dictionary contain the results of crop segmentation and the map.

    Version 0 Only support 2C-classification
    Data: Feb 14 2022
    Author: Huajian Liu
    """

    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import transformation as tf
    from appf_toolbox.hyper_processing import pre_processing as pp
    from matplotlib import pyplot as plt
    import joblib as joblib
    import numpy as np
    from skimage import exposure

    # ------------------------------------------------------------------------------------------------------------------
    # Crop segmentation
    # ------------------------------------------------------------------------------------------------------------------
    print('Conducting crop segmentation......')
    bw_crop, pseu_crop = pp.green_plant_segmentation(hyp_data,
                                                     wavelength,
                                                     seg_model_path,
                                                     seg_model_name,
                                                     band_R=band_r,
                                                     band_G=band_g,
                                                     band_B=band_b,
                                                     gamma=gamma,
                                                     flag_remove_noise=flag_remove_noise,
                                                     flag_check=flag_figure)

    # ------------------------------------------------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------------------------------------------------
    print('Creating maps......')
    # Load model
    classification_model = joblib.load(classification_model_path + '/' + classification_model_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------------------------------------------------------
    # Save RGB for showing results
    r = hyp_data[:, :, band_r].copy()
    g = hyp_data[:, :, band_g].copy()
    b = hyp_data[:, :, band_b].copy()
    wave_r = wavelength[band_r]
    wave_g = wavelength[band_g]
    wave_b = wavelength[band_b]

    # Smooth
    (n_row, n_col, _) = hyp_data.shape
    hyp_data = hyp_data.reshape(n_row * n_col, hyp_data.shape[2], order='c')
    hyp_data = pp.smooth_savgol_filter(hyp_data,
                                       window_length=classification_model['Note']['smooth_window_length'],
                                       polyorder=classification_model['Note']['smooth_window_polyorder'])
    hyp_data[hyp_data < 0] = 0

    # Resampling
    hyp_data = pp.spectral_resample(hyp_data, wavelength, classification_model['Note']['wavelengths used in the model'])

    # ------------------------------------------------------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------------------------------------------------------
    if classification_model['Note']['data_transformation'] == 'hyp-hue':
        # hc2hhsi
        hyp_data, _, _ = tf.hc2hhsi(hyp_data.reshape(n_row, n_col, hyp_data.shape[1]))
        hyp_data = hyp_data.reshape(hyp_data.shape[0] * hyp_data.shape[1], hyp_data.shape[2])
    else:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Make map
    # ------------------------------------------------------------------------------------------------------------------
    # Predict
    bw_cl = classification_model['final model'].predict(hyp_data) # 1D
    bw_cl = bw_cl.reshape(n_row, n_col) # 2D
    bw_cl = bw_cl.astype(bool)
    bw_cl[bw_crop==False] = False

    # Make a pseudo image of map
    r[bw_cl == True] = 1
    g[bw_cl == True] = 0
    b[bw_cl == True] = 0
    pseu_cl = np.concatenate((r.reshape(n_row, n_col, 1),
                              g.reshape(n_row, n_col, 1),
                              b.reshape(n_row, n_col, 1)),
                             axis=2)

    pseu_cl = exposure.adjust_gamma(pseu_cl, gamma)

    # Show result
    if flag_figure:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Classification map')
        ax[0].imshow(bw_cl, cmap='gray')
        ax[0].set_title('BW of classification')
        ax[1].imshow(pseu_cl)
        ax[1].set_title('Pseudo RGB R=' + str(wave_r) + ' G=' + str(wave_g) + ' B=' + str(wave_b))
        plt.show()

    print('All done!')

    return {'bw_crop': bw_crop,
            'pseu_crop': pseu_crop,
            'bw_classification': bw_cl,
            'pseu_classification': pseu_cl}


# ########################################################################################################################
# # Demo of make_class_map
# ########################################################################################################################
# if __name__ == "__main__":
#     import sys
#     import numpy as np
#     sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
#     from appf_toolbox.hyper_processing import envi_funs
#
#     # Input parameters
#     data_path = '/media/huajian/Files/Data/crown_rot_0590/wiw_top_view_examples'
#     data_name = 'vnir_100_140_9064_2021-08-10_01-08-07'
#
#     seg_model_path = '/media/huajian/Files/python_projects/p12_green_segmentation/green_seg_model_20220201'
#     seg_model_name = 'record_OneClassSVM_vnir_hh.sav'
#
#     classification_model_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/' \
#                         'crown_rot_0590_top_view/2c_models'
#     classification_model_name = 'model_svm_2c_vnir_hyp-hue_top_view_2021-08-03.sav'
#
#     # Read the data
#     raw_data, meta_plant = envi_funs.read_hyper_data(data_path, data_name)
#
#     wavelengths = np.zeros((meta_plant.metadata['Wavelength'].__len__(),))
#     for i in range(wavelengths.size):
#         wavelengths[i] = float(meta_plant.metadata['Wavelength'][i])
#
#     # Calibrate the data
#     hypcube = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'],
#                                              trim_rate_t_w=0.1, trim_rate_b_w=0.95)
#
#     # Make map
#     map = make_class_map(hypcube,
#                          wavelengths,
#                          seg_model_path,
#                          seg_model_name,
#                          classification_model_path,
#                          classification_model_name,
#                          band_r=97,
#                          band_g=52,
#                          band_b=14,
#                          gamma=0.7,
#                          flag_remove_noise=True,
#                          flag_figure=True)

