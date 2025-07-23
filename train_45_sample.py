import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from classification_HL import tune_svm_classification_YX
import numpy as np
from tqdm import tqdm

def svm_repeated_runs(excel_path, train_num, test_num_list, run_number, output_excel):
    df = pd.read_excel(excel_path)

    all_ids = df['Plant ID'].unique()
    results = []
    np.random.seed(2024)

    for run in tqdm(range(run_number), desc="Running SVM trials"):
        # 随机打乱ID顺序
        shuffled_ids = np.random.permutation(all_ids)

        train_ids = shuffled_ids[:train_num]
        remaining_ids = [pid for pid in shuffled_ids if pid not in train_ids]

        train_df = df[df['Plant ID'].isin(train_ids)].copy()

        X_train = train_df.iloc[:, 4:-2].values
        y_train = train_df.iloc[:, -1].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        mapping = {'ZS37': 0, 'ZS39': 1, 'ZS41': 2}
        y_train = np.array([mapping.get(item, item) for item in y_train])

        # SVM
        opt_num_iter_cv = 5
        opt_num_fold_cv = 5
        opt_num_evals = 3
        svm_kernel_options = ['linear', 'rbf', 'poly']
        svm_c_range = [1, 100]
        svm_gamma_range = [0.01, 50]
        svm_coef0_range = [0, 50]
        svm_tol = 1e-3

        # train
        best_model = tune_svm_classification_YX(
            X_train_scaled, y_train, svm_kernel_options, svm_c_range,
            svm_gamma_range, svm_coef0_range, svm_tol,
            opt_num_iter_cv, opt_num_fold_cv, opt_num_evals
        )

        run_result = {'Run': run + 1}
        for test_num in test_num_list:
            # 确保test_num不会超过剩下的ID数
            if test_num > len(remaining_ids):
                continue

            selected_test_ids = np.random.choice(remaining_ids, size=test_num, replace=False)
            test_df = df[df['Plant ID'].isin(selected_test_ids)].copy()

            X_test = test_df.iloc[:, 4:-2].values
            y_test = test_df.iloc[:, -1].values
            X_test_scaled = scaler.transform(X_test)
            y_test = np.array([mapping.get(item, item) for item in y_test])

            y_pred = best_model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, average='weighted')
            run_result[f'F1_{test_num}'] = f1

        results.append(run_result)

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_excel, index=False)
    print(f"F1 scores saved to {output_excel}")

output_excel = 'svm_f1_scores_multi_test_new.xlsx'

svm_repeated_runs(
    excel_path=r"I:\paper 2 analyse\ASD dataset\45_min\full_data.xlsx",
    train_num=45,
    test_num_list=[45, 90, 135, 180, 225],
    run_number=1000,
    output_excel=output_excel
)
