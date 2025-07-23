from classification_HL import *
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_shannon_diversity(train_data, num_bins=10):
    """
    Calculate the Shannon diversity index for a given train_data.

    Args:
    train_data (numpy.ndarray or pandas.DataFrame): The spectral data where rows are samples and columns are features (e.g., wavelengths).
    num_bins (int): The number of bins for histogram calculation.

    Returns:
    float: The mean Shannon diversity index across all features.
    """
    # Ensure train_data is a numpy array
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values

    # Initialize a list to store Shannon diversity values for each column
    shannon_values = []

    # Calculate Shannon diversity for each column
    for i in range(train_data.shape[1]):  # Iterate over each feature (column)
        column_data = train_data[:, i]

        # Compute histogram
        hist, _ = np.histogram(column_data, bins=num_bins, range=(column_data.min(), column_data.max()))

        # Calculate frequencies
        probabilities = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)

        # Calculate Shannon diversity for this column
        shannon_index = -np.sum(
            probabilities * np.log(probabilities + np.finfo(float).eps))  # Add epsilon to avoid log(0)
        shannon_values.append(shannon_index)

    # Return the mean Shannon diversity index across all columns
    return np.mean(shannon_values)


def calculate_simpson_diversity(train_data, num_bins=10):
    """
    Calculate the Simpson diversity index for a given train_data.

    Args:
    train_data (numpy.ndarray or pandas.DataFrame): The spectral data where rows are samples and columns are features (e.g., wavelengths).
    num_bins (int): The number of bins for histogram calculation.

    Returns:
    float: The mean Simpson diversity index across all features.
    """
    # Ensure train_data is a numpy array
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values

    # Initialize a list to store Simpson diversity values for each column
    simpson_values = []

    # Calculate Simpson diversity for each column
    for i in range(train_data.shape[1]):  # Iterate over each feature (column)
        column_data = train_data[:, i]

        # Compute histogram
        hist, _ = np.histogram(column_data, bins=num_bins, range=(column_data.min(), column_data.max()))

        # Calculate frequencies
        probabilities = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)

        # Calculate Simpson diversity for this column
        simpson_index = 1 - np.sum(probabilities ** 2)  # Simpson index formula
        simpson_values.append(simpson_index)

    # Return the mean Simpson diversity index across all columns
    return np.mean(simpson_values)


# 设置随机种子
np.random.seed(2024)

train_data_path = r"I:\paper 2 analyse\ASD dataset\train_data_average_early.xlsx"
test_data_path = r"I:\paper 2 analyse\ASD dataset\test_data_average_early.xlsx"
save_dir = r'I:\paper 2 analyse\ASD result\min_num\early\snv'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取数据
df_train = pd.read_excel(train_data_path, sheet_name='snv')
df_test = pd.read_excel(test_data_path,sheet_name='snv')
selected_num = 10000
mapping = {'ZS37': 0, 'ZS39': 1, 'ZS41': 2}
df_train.iloc[:, -1] = df_train.iloc[:, -1].map(mapping)
df_test.iloc[:, -1] = df_test.iloc[:, -1].map(mapping)

print('Data loading finished!')

selected_plant_id_num_list = [10,20, 30, 40, 50, 60]

for selected_plant_id_num in selected_plant_id_num_list:
    print('Selected plant id number: ' + str(selected_plant_id_num))
    if not os.path.exists(os.path.join(save_dir,str(selected_plant_id_num))):
        os.makedirs(os.path.join(save_dir,str(selected_plant_id_num)))

    excel_path = os.path.join(save_dir, str(selected_plant_id_num),'diversity_and_distribution_measures_full.xlsx')

    plant_id_column = df_train.columns[-2]


    used_combinations = set()
    total_selected_ids = []
    total_F1_score = []
    save_f1_ZS37 = []
    save_f1_ZS39 = []
    save_f1_ZS41 = []


    for _ in tqdm(range(selected_num), desc="Processing diversity measures"):

        new_combo = tuple()
        while new_combo in used_combinations or not new_combo:
            selected_ids = np.random.choice(df_train[plant_id_column].unique(), size=selected_plant_id_num, replace=False)
            new_combo = tuple(sorted(selected_ids))
        used_combinations.add(new_combo)
        total_selected_ids.append(selected_ids)

        train_dataset = df_train[df_train[plant_id_column].isin(selected_ids)]

        # 提取数据
        train_data = train_dataset.iloc[:, 4:-2].values
        test_data = df_test.iloc[:, 4:-2].values

        X_train = train_data  # 特征
        X_test = test_data
        y_train = train_dataset.iloc[:, -1].values


        y_test = df_test.iloc[:, -1].values

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        opt_num_iter_cv = 5
        opt_num_fold_cv = 5
        opt_num_evals = 3

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

        # 执行重复K折交叉验证并保存模型
        best_model = tune_svm_classification_YX(X_train, y_train, svm_kernel_options, svm_c_range,
                                                svm_gamma_range, svm_coef0_range, svm_tol, opt_num_iter_cv,
                                                opt_num_fold_cv,
                                                opt_num_evals)


        y_pred = best_model.predict(X_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True)


        test_f1_score = report_dict['weighted avg']['f1-score']

        total_F1_score.append(test_f1_score)


        f1_ZS37 = report_dict['0']['f1-score']
        f1_ZS39 = report_dict['1']['f1-score']
        f1_ZS41 = report_dict['2']['f1-score']

        save_f1_ZS37.append(f1_ZS37)
        save_f1_ZS39.append(f1_ZS39)
        save_f1_ZS41.append(f1_ZS41)



    data = {
        "Selected_IDs": total_selected_ids,
        "F1_Score": total_F1_score,
        "F1_ZS37": save_f1_ZS37,
        "F1_ZS39": save_f1_ZS39,
        "F1_ZS41": save_f1_ZS41
    }

    df_to_save = pd.DataFrame(data)

    df_to_save.to_excel(excel_path, index=False)

    print(f"Data saved to {excel_path}.")

    print('Selected plant id number: ' + str(selected_plant_id_num) + ' finished')




