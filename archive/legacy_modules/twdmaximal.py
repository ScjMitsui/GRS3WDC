import pandas as pd
import numpy as np
import math
import networkx as nx


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances, precision_score, recall_score, f1_score, accuracy_score
from joblib import Parallel, delayed

def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

def find_maximal_cliques(adj):
    # 创建图
    G = nx.Graph(adj)
    # 找到所有极大团
    cliques = list(nx.find_cliques(G))
    return cliques

def run_model(X_train, y_train, X_test, y_test, delta, alpha):
    n, m = X_train.shape

    # 使用向量化计算距离矩阵
    dis_arr = pairwise_distances(X_train, metric='euclidean')
    adj_arr = (dis_arr < delta).astype(int)

    # 找到所有极大团
    cliques = find_maximal_cliques(adj_arr)

    # 并行过滤符合条件的团
    def filter_clique(clique):
        temp_label = y_train[clique]
        count = pd.Series(temp_label).value_counts()
        if not count[count > alpha * len(clique)].empty:
            label = count[count > alpha * len(clique)].index[0]
            return clique, label
        else:
            return None

    filtered_results = Parallel(n_jobs=-1)(delayed(filter_clique)(clique) for clique in cliques)

    filtered_cliques = []
    cliques_label = []

    for result in filtered_results:
        if result is not None:
            clique, label = result
            filtered_cliques.append(clique)
            cliques_label.append(label)

    # 初始化变量以收集预测结果和真实标签
    y_true = []
    y_pred = []

    # 并行处理测试样本
    def process_test_sample(i, t):
        mc_labels = []
        for k, clique in enumerate(filtered_cliques):
            # 检查测试样本与团中所有样本的距离是否都小于 delta
            distances = np.linalg.norm(X_train[clique] - t, axis=1)
            if np.all(distances < delta):
                mc_labels.append(cliques_label[k])
        if mc_labels:
            if all(label == mc_labels[0] for label in mc_labels):
                predicted_label = mc_labels[0]
                return y_test[i], predicted_label
            else:
                # 标签不一致，无法预测
                return y_test[i], None
        else:
            return y_test[i], None  # 无法预测时，返回 None

    test_results = Parallel(n_jobs=-1)(delayed(process_test_sample)(i, t) for i, t in enumerate(X_test))

    # 收集预测结果和真实标签
    for true_label, pred_label in test_results:
        y_true.append(true_label)
        y_pred.append(pred_label)

    # 过滤掉无法预测的样本
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
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        coverage = 0

    # 计算覆盖率

    return accuracy, precision, recall, f1, coverage