import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
from classification_HL import *


import pandas as pd
import itertools

def load_sheets_data(excel_path, sheet_names):
    """读取并返回指定工作表的数据字典，保留每个工作表的最后两列，并只合并指定的列。"""
    data = {}
    for sheet_name in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # 提取合并的列范围和最后两列
        merge_cols = df.iloc[:, 2:-2]  # 从第五列到倒数第三列
        last_two_cols = df.iloc[:, -2:]  # 最后两列
        data[sheet_name] = (merge_cols, last_two_cols)
    return data

def combine_sheets_data(data):
    """生成所有可能的工作表组合及其数据，只保留一份最后两列。"""
    combined_data = {}
    # 单独的工作表
    for sheet_name, (merge_cols, last_two_cols) in data.items():
        combined_data[sheet_name] = pd.concat([merge_cols, last_two_cols], axis=1)

    # 工作表组合
    for r in range(2, len(data) + 1):
        for combo in itertools.combinations(data.keys(), r):
            # 获取每个sheet的合并列数据副本
            combined_dfs = [data[sheet][0].copy() for sheet in combo]
            # 将所有合并列数据合并
            combined_df = pd.concat(combined_dfs, axis=1)
            # 选择一个工作表的最后两列添加到合并数据中
            last_two_cols = data[combo[0]][1].copy()  # 选择第一个工作表的最后两列
            # 合并所有列和最后两列
            combined_df = pd.concat([combined_df, last_two_cols], axis=1)
            combined_df.columns = list(range(combined_df.shape[1]))  # 重新编号列为0, 1, 2,...
            combined_data['+'.join(combo)] = combined_df

    return combined_data

def get_data_combinations(excel_path, sheet_names):
    """主函数：加载 Excel 数据，合并工作表，并返回组合数据字典。"""
    # 加载数据
    data = load_sheets_data(excel_path, sheet_names)
    # 生成组合数据
    combined_data = combine_sheets_data(data)
    return combined_data



# 示例使用
excel_path = r"I:\0757 hyperspectral\labelled_data_0757 - three.xlsx"
sheet_names = ['org', 'hyp', 'snv', 'pca']
model_save_dir = 'three_class_xgboost'
model_type = 'xgboost' #rf, nn, svm
data_combinations = get_data_combinations(excel_path, sheet_names)
combo_key = data_combinations.keys()

print(combo_key)

save_key = []
save_train_f1 = []
save_val_f1 = []
save_test_f1 = []

save_f1_ZS37 =[]
save_f1_ZS39 = []
save_f1_ZS41 = []


excel_path = 'train_test_ids_two.xlsx'  # 修改为你的Excel文件路径
id_data = pd.read_excel(excel_path)

for key in combo_key:
    save_key.append(key)

    data = data_combinations[key]

    # 数据预处理
    X = data.iloc[:, :-2].values  # 特征
    y = data.iloc[:, -1].values  # 标签

    # 读取Excel文件


    # 从Excel文件中提取train_ids和test_ids
    train_ids = id_data['train_ids'].dropna().values  # 删除任何可能的NaN值
    test_ids = id_data['test_ids'].dropna().values  # 删除任何可能的NaN值

    # 从原数据中获取id_tags
    id_tags = data.iloc[:, -2].values  # 假设id_tags是数据中的倒数第二列

    # 根据Excel文件中的ids创建掩码
    train_mask = np.isin(id_tags, train_ids)
    test_mask = np.isin(id_tags, test_ids)

    # 使用掩码划分数据
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    mapping = {'ZS37': 0, 'ZS39': 1, 'ZS41': 2}
    y_train = np.array([mapping.get(item, item) for item in y_train])
    y_test = np.array([mapping.get(item, item) for item in y_test])

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

    # svm_kernel = 'rbf'
    # svm_c_range = [1, 100]
    # svm_gamma_range = [0.01, 50]
    # svm_tol = 1e-3
    #
    # karg_tune_model = {'svm_kernel': svm_kernel,
    #                    'svm_c_range': svm_c_range,
    #                    'svm_gamma_range': svm_gamma_range,
    #                    'svm_tol': svm_tol,
    #                    'opt_num_iter_cv': opt_num_iter_cv,
    #                    'opt_num_fold_cv': opt_num_fold_cv,
    #                    'opt_num_evals': opt_num_evals}
    #
    # result, file_name_save, best_model = repeadted_kfold_cv(X_train, y_train, key, model_save_dir,
    #                                                           n_splits=rkf_n_splits,
    #                                                           n_repeats=rkf_n_repeats,
    #                                                           tune_model=tune_svm_classification,
    #                                                           karg=karg_tune_model,
    #                                                           random_state=rkf_random_state,flag_save=True)
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

        # 执行重复K折交叉验证并保存模型
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

        # 将所有参数整合为 karg_tune_model 格式
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

        # 执行重复K折交叉验证并保存模型
        result, file_name_save, best_model = repeadted_kfold_cv(X_train, y_train, key, model_save_dir,
                                                                n_splits=5,
                                                                n_repeats=3,
                                                                tune_model=tune_nn_classification,
                                                                karg=karg_tune_model,
                                                                random_state=rkf_random_state,
                                                                flag_save=True)

    elif model_type == 'xgboost':
        # 将所有参数整合为 karg_tune_model 格式
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
                                                                flag_save=True,xgboost=True)


    print('----------------------------------------------------------------------------------------------')
    if not model_type == 'xgboost':
        print('Best model: ' + str(best_model.get_params()))

    print('F1 summary')
    ave_metrics = result['average metrics']  # 获取平均指标
    train_metrics = ave_metrics['ave_metrics_train']  # 获取训练指标

    # 从训练指标中获取 F1 分数
    train_f1_score = train_metrics['f1']

    save_train_f1.append(train_f1_score)

    # 打印 F1 分数
    print("Train F1 Score:", train_f1_score)

    validation_metrics = ave_metrics['ave_metrics_validation']  # 获取验证指标

    # 从验证指标中获取 F1 分数
    val_f1_score = validation_metrics['f1']

    save_val_f1.append(val_f1_score)

    # 打印 F1 分数
    print("Validation F1 Score:", val_f1_score)

    print('----------------------------------------------------------------------------------------------')
    # 加载训练好的模型
    loaded_model = joblib.load(file_name_save)

    from sklearn.metrics import classification_report

    # 检查模型结构并预测
    if isinstance(loaded_model, dict) and 'final model' in loaded_model:
        if model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_pred = best_model.predict(dtest)
        else:
            model = loaded_model['final model']
            y_pred = model.predict(X_test)

        # 生成分类报告的字典形式
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # 提取加权平均 F1 分数
        test_f1_score = report_dict['weighted avg']['f1-score']

        save_test_f1.append(test_f1_score)

        print("Classification Report on Test Data:")
        print(classification_report(y_test, y_pred))
        print("Test F1 Score:", test_f1_score)

        # 提取特定类别的 F1 分数
        f1_ZS37 = report_dict['0']['f1-score']
        f1_ZS39 = report_dict['1']['f1-score']
        f1_ZS41 = report_dict['2']['f1-score']

        save_f1_ZS37.append(f1_ZS37)
        save_f1_ZS39.append(f1_ZS39)
        save_f1_ZS41.append(f1_ZS41)

        print("Classification Report on Test Data:")
        print("F1 Score for ZS37:", f1_ZS37)
        print("F1 Score for ZS39:", f1_ZS39)
        print("F1 Score for ZS41:", f1_ZS41)
    else:
        print("Loaded data does not contain a valid model.")


# 创建字典，字典的键为列名，值为数据列表
final_data = {
    'Combo': save_key,
    'Train F1': save_train_f1,
    'Validation F1': save_val_f1,
    'Test F1': save_test_f1,
    'F1 ZS37': save_f1_ZS37,
    'F1 ZS39': save_f1_ZS39,
    'F1 ZS41': save_f1_ZS41
}

# 创建 DataFrame
final_df = pd.DataFrame(final_data)

# 保存 DataFrame 到 Excel 文件
output_excel_path = model_save_dir + '/' + 'final_result-three_svm.xlsx'
final_df.to_excel(output_excel_path, index=False)  # index=False 以避免添加额外的索引列

