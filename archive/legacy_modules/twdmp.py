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

def run_model(X_train, y_train, X_test, y_test, beta, alpha):
    n, m = X_train.shape

    # 使用向量化计算距离矩阵
    dis_arr = pairwise_distances(X_train, metric='euclidean')
    distances = dis_arr[np.triu_indices_from(dis_arr, k=1)]
    delta = np.percentile(distances, beta)
    adj_arr = (dis_arr < delta).astype(int)

    # 找到所有极大团
    cliques = find_maximal_cliques(adj_arr)
    print('Found!', len(cliques))

    # 并行过滤符合条件的团
    def filter_clique(clique):
        temp_label = y_train[clique]
        count = pd.Series(temp_label).value_counts()
        if not count[count > alpha * len(clique)].empty:
            label = count[count > alpha * len(clique)].index[0]
            return clique, label
        else:
            return None

    filtered_results = Parallel(n_jobs=3)(delayed(filter_clique)(clique) for clique in cliques)

    filtered_cliques = []
    cliques_label = []

    filtered_non_none = [res for res in filtered_results if res is not None]
    filtered_cliques, cliques_label = zip(*filtered_non_none)
    filtered_cliques = list(filtered_cliques)
    cliques_label = list(cliques_label)

    print('Filtered!', len(filtered_cliques))

    # 初始化变量以收集预测结果和真实标签
    y_true = []
    y_pred = []

    # 并行处理测试样本
    # 并行处理测试样本
    def process_test_sample(i, t):
        # 预计算测试样本 t 到所有训练样本的距离
        distances = np.linalg.norm(X_train - t, axis=1)
        
        # 使用列表生成式过滤符合条件的 cliques_label
        mc_labels = {cliques_label[k] for k, clique in enumerate(filtered_cliques) if np.all(distances[clique] < delta)}
        
        # 判断集合中是否只有一个唯一的标签
        if len(mc_labels) == 1:
            predicted_label = next(iter(mc_labels))
            return y_test[i], predicted_label
        else:
            return y_test[i], None  # 无法预测时，返回 None

    test_results = Parallel(n_jobs=3)(delayed(process_test_sample)(i, t) for i, t in enumerate(X_test))

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