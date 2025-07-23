import joblib

from classification_HL import *

from sklearn.preprocessing import StandardScaler
import pandas as pd
import itertools

def load_sheets_data(excel_path, sheet_names):
    """Read and return data from specified sheets as a dictionary. Only keep merged columns and the last two columns."""
    data = {}
    for sheet_name in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        merge_cols = df.iloc[:, 4:-2]  # Columns from 5th to the third last
        last_two_cols = df.iloc[:, -2:]  # Last two columns
        data[sheet_name] = (merge_cols, last_two_cols)
    return data

def combine_sheets_data(data):
    """Generate all possible sheet combinations with merged features and one copy of the last two columns."""
    combined_data = {}
    # Single sheet
    for sheet_name, (merge_cols, last_two_cols) in data.items():
        combined_data[sheet_name] = pd.concat([merge_cols, last_two_cols], axis=1)

    # Sheet combinations
    for r in range(2, len(data) + 1):
        for combo in itertools.combinations(data.keys(), r):
            combined_dfs = [data[sheet][0].copy() for sheet in combo]
            combined_df = pd.concat(combined_dfs, axis=1)
            last_two_cols = data[combo[0]][1].copy()
            combined_df = pd.concat([combined_df, last_two_cols], axis=1)
            combined_df.columns = list(range(combined_df.shape[1]))  # Rename columns to 0,1,2,...
            combined_data['+'.join(combo)] = combined_df

    return combined_data

def get_data_combinations(excel_path, sheet_names):
    """Main function to load Excel sheets and return all combined datasets."""
    data = load_sheets_data(excel_path, sheet_names)
    combined_data = combine_sheets_data(data)
    return combined_data

excel_path_list = [
                   r"I:\paper 2 analyse\ASD dataset\train_data_average.xlsx"]
excel_path_test_list = [
                        r"I:\paper 2 analyse\ASD dataset\test_data_average.xlsx"]
model_save_dir_list = [r'I:\paper 2 analyse\ASD result\full_svm']

sheet_names = ['org', 'hyp', 'snv', 'pca']

for i in range(1):
    excel_path = excel_path_list[i]
    excel_path_test = excel_path_test_list[i]
    model_save_dir = model_save_dir_list[i]

    print(str(excel_path) + ' processing')
    print(str(excel_path_test) + ' test processing')
    print(str(model_save_dir) + ' saving')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_type = 'svm'  # options: rf, nn, svm
    data_combinations = get_data_combinations(excel_path, sheet_names)
    combo_key = data_combinations.keys()

    data_combinations_test = get_data_combinations(excel_path_test, sheet_names)

    save_key = []
    save_train_f1 = []
    save_val_f1 = []
    save_test_f1 = []

    save_f1_ZS37 = []
    save_f1_ZS39 = []
    save_f1_ZS41 = []
    scaler = StandardScaler()

    for key in combo_key:
        save_key.append(key)

        data = data_combinations[key]
        data_test = data_combinations_test[key]

        # Data preprocessing
        X_train = data.iloc[:, :-2].values
        y_train = data.iloc[:, -1].values

        X_test = data_test.iloc[:, :-2].values
        y_test = data_test.iloc[:, -1].values

        mapping = {'ZS37': 0, 'ZS39': 1, 'ZS41': 2}
        y_train = np.array([mapping.get(item, item) for item in y_train])
        y_test = np.array([mapping.get(item, item) for item in y_test])

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rkf_n_splits = 5
        rkf_n_repeats = 3
        rkf_random_state = 2024

        opt_num_iter_cv = 5
        opt_num_fold_cv = 5
        opt_num_evals = 3

        if model_type == 'rf':
            rf_n_estimators_range = [20, 200]
            rf_max_depth_range = [2, 50]
            rf_min_samples_split_range = [2, 10]
            rf_min_samples_leaf_range = [1, 5]

            karg_tune_model_rf = {
                'rf_n_estimators_range': rf_n_estimators_range,
                'rf_max_depth_range': rf_max_depth_range,
                'rf_min_samples_split_range': rf_min_samples_split_range,
                'rf_min_samples_leaf_range': rf_min_samples_leaf_range,
                'opt_num_iter_cv': opt_num_iter_cv,
                'opt_num_fold_cv': opt_num_fold_cv,
                'opt_num_evals': opt_num_evals
            }
            result, file_name_save, best_model = repeadted_kfold_cv(X_train, y_train, key, model_save_dir,
                                                                    n_splits=rkf_n_splits,
                                                                    n_repeats=rkf_n_repeats,
                                                                    tune_model=tune_random_forest_classification,
                                                                    karg=karg_tune_model_rf,
                                                                    random_state=rkf_random_state, flag_save=True)

        elif model_type == 'svm':
            svm_kernel_options = ['linear', 'rbf', 'poly']
            svm_c_range = [1, 100]
            svm_gamma_range = [0.01, 50]
            svm_coef0_range = [0, 50]
            svm_tol = 1e-3

            karg_tune_model = {'svm_kernel_options': svm_kernel_options,
                               'svm_c_range': svm_c_range,
                               'svm_gamma_range': svm_gamma_range,
                               'svm_tol': svm_tol,
                               'svm_coef0_range': svm_coef0_range,
                               'opt_num_iter_cv': opt_num_iter_cv,
                               'opt_num_fold_cv': opt_num_fold_cv,
                               'opt_num_evals': opt_num_evals}

            result, file_name_save, best_model = repeadted_kfold_cv(X_train, y_train, key, model_save_dir,
                                                                    n_splits=5,
                                                                    n_repeats=3,
                                                                    tune_model=tune_svm_classification_YX,
                                                                    karg=karg_tune_model,
                                                                    random_state=rkf_random_state,
                                                                    flag_save=True)

        elif model_type == 'nn':
            hidden_layer_sizes_range = [(10,), (50,), (100,), (100, 50), (100, 10), (50, 10), (100, 50, 10)]
            activation_options = ['relu', 'tanh']
            alpha_range = [0.0001, 0.01]
            learning_rate_init_range = [0.001, 0.1]
            solver_options = ['adam', 'sgd']

            karg_tune_model = {
                'hidden_layer_sizes_range': hidden_layer_sizes_range,
                'activation_options': activation_options,
                'alpha_range': alpha_range,
                'learning_rate_init_range': learning_rate_init_range,
                'solver_options': solver_options,
                'opt_num_iter_cv': opt_num_iter_cv,
                'opt_num_fold_cv': opt_num_fold_cv,
                'opt_num_evals': opt_num_evals
            }

            result, file_name_save, best_model = repeadted_kfold_cv(X_train, y_train, key, model_save_dir,
                                                                    n_splits=5,
                                                                    n_repeats=3,
                                                                    tune_model=tune_nn_classification,
                                                                    karg=karg_tune_model,
                                                                    random_state=rkf_random_state,
                                                                    flag_save=True)

        elif model_type == 'xgboost':
            opt_num_iter_cv = 3
            karg_tune_model = {
                'opt_num_iter_cv': opt_num_iter_cv,
                'opt_num_fold_cv': opt_num_fold_cv,
                'opt_num_evals': opt_num_evals
            }

            print(model_type)

            result, file_name_save, best_model = repeadted_kfold_cv(X_train, y_train, key, model_save_dir,
                                                                    n_splits=5,
                                                                    n_repeats=3,
                                                                    tune_model=tune_xgboost_YX,
                                                                    karg=karg_tune_model,
                                                                    random_state=rkf_random_state,
                                                                    flag_save=True, xgboost=True)

        print('----------------------------------------------------------------------------------------------')
        if not model_type == 'xgboost':
            print('Best model: ' + str(best_model.get_params()))

        print('F1 summary')
        ave_metrics = result['average metrics']
        train_metrics = ave_metrics['ave_metrics_train']
        train_f1_score = train_metrics['f1']
        save_train_f1.append(train_f1_score)
        print("Train F1 Score:", train_f1_score)

        validation_metrics = ave_metrics['ave_metrics_validation']
        val_f1_score = validation_metrics['f1']
        save_val_f1.append(val_f1_score)
        print("Validation F1 Score:", val_f1_score)

        print('----------------------------------------------------------------------------------------------')

        loaded_model = joblib.load(file_name_save)

        from sklearn.metrics import classification_report

        if isinstance(loaded_model, dict) and 'final model' in loaded_model:
            if model_type == 'xgboost':
                dtest = xgb.DMatrix(X_test, label=y_test)
                y_pred = best_model.predict(dtest)
            else:
                model = loaded_model['final model']
                y_pred = model.predict(X_test)

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            test_f1_score = report_dict['weighted avg']['f1-score']
            save_test_f1.append(test_f1_score)

            print("Classification Report on Test Data:")
            print(classification_report(y_test, y_pred))
            print("Test F1 Score:", test_f1_score)

            f1_ZS37 = report_dict['0']['f1-score']
            f1_ZS39 = report_dict['1']['f1-score']
            f1_ZS41 = report_dict['2']['f1-score']

            save_f1_ZS37.append(f1_ZS37)
            save_f1_ZS39.append(f1_ZS39)
            save_f1_ZS41.append(f1_ZS41)

            print("F1 Score for ZS37:", f1_ZS37)
            print("F1 Score for ZS39:", f1_ZS39)
            print("F1 Score for ZS41:", f1_ZS41)
        else:
            print("Loaded data does not contain a valid model.")

    final_data = {
        'Combo': save_key,
        'Train F1': save_train_f1,
        'Validation F1': save_val_f1,
        'Test F1': save_test_f1,
        'F1 ZS37': save_f1_ZS37,
        'F1 ZS39': save_f1_ZS39,
        'F1 ZS41': save_f1_ZS41
    }

    final_df = pd.DataFrame(final_data)

    output_excel_path = model_save_dir + '/' + 'final_result-three_ASD.xlsx'
    final_df.to_excel(output_excel_path, index=False)
