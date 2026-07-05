import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances, precision_score, recall_score, f1_score, accuracy_score
from joblib import Parallel, delayed

def chebyshev_distance(x, y):
    return np.max(np.abs(x - y), axis=1)

def run_model(X_train, y_train, X_test, y_test, beta, alpha):
    n, m = X_train.shape

    # 使用向量化计算距离矩阵，切比雪夫距离
    dis_arr = pairwise_distances(X_train, metric='chebyshev')
    distances = dis_arr[np.triu_indices_from(dis_arr, k=1)]
    delta = np.percentile(distances, beta)

    # 初始化结果列表
    neighborhoods = []
    neighborhoods_label = []
    granules = []

    # 并行处理每个训练样本
    def process_training_sample(i):
        # 找到与样本 i 距离小于 delta 的所有样本索引
        neighbors = np.where(dis_arr[i] < delta)[0]
        temp_label = y_train[neighbors]
        count = pd.Series(temp_label).value_counts()
        if not count[count > alpha * len(neighbors)].empty:
            label = count[count > alpha * len(neighbors)].index.tolist()[0]
            return i, neighbors, label
        else:
            return None

    # 使用多线程并行处理
    results = Parallel(n_jobs=5)(delayed(process_training_sample)(i) for i in range(n))

    # 过滤结果
    for res in results:
        if res is not None:
            i, neighbors, label = res
            neighborhoods.append(i)
            granules.append(neighbors)
            neighborhoods_label.append(label)

    # 将 neighborhoods 转换为 NumPy 数组以便于索引
    neighborhoods = np.array(neighborhoods)

    # 初始化变量以收集预测结果和真实标签
    y_true = []
    y_pred = []

    # 并行处理测试样本
    def process_test_sample(i, t):
        # 计算测试样本与所有训练样本的切比雪夫距离
        dis_to_train = np.max(np.abs(X_train - t), axis=1)
        # 找到距离小于 delta 的训练样本索引
        mask = dis_to_train[neighborhoods] < delta
        if np.any(mask):
            close_indices = neighborhoods[mask]
            mc_labels = y_train[close_indices].tolist()
            if len(mc_labels) > 0:
                # 检查 mc_labels 中的所有标签是否一致
                if all(label == mc_labels[0] for label in mc_labels):
                    predicted_label = mc_labels[0]
                    return y_test[i], predicted_label
                else:
                    # 标签不一致，无法预测
                    return y_test[i], None
            else:
                return y_test[i], None  # 无法预测时，返回 None
        else:
            return y_test[i], None

    # 使用多线程并行处理测试样本
    test_results = Parallel(n_jobs=5)(delayed(process_test_sample)(i, t) for i, t in enumerate(X_test))

    # 收集预测结果和真实标签
    for true_label, pred_label in test_results:
        y_true.append(true_label)
        y_pred.append(pred_label)

    # 过滤无法预测的样本
    y_true_filtered = []
    y_pred_filtered = []
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label is not None:
            y_true_filtered.append(true_label)
            y_pred_filtered.append(pred_label)

    # 计算评价指标
    if y_pred_filtered:
        coverage = len(y_pred_filtered) / len(y_test)
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        precision = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        recall = accuracy * coverage
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        coverage = 0

    # 返回计算的评价指标
    return accuracy, precision, recall, f1, coverage