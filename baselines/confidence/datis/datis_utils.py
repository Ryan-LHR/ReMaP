# -*-coding:utf-8-*-
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer

from utils import save_to_pickle, load_from_pickle, save_point_print


def find_closest_ratio(target_ratio, ratio_list):
    """
    找到ratio_list中最接近target_ratio的比例的索引。

    Args:
        target_ratio (float): 目标比例。
        ratio_list (list of float): 预定义的比例列表。

    Returns:
        int: 最接近比例的索引。
    """
    closest_ratio = min(ratio_list, key=lambda x: abs(x - target_ratio))
    return ratio_list.index(closest_ratio)


def DATIS_test_input_selection(softmax_prob, train_support_output,
                               y_train, test_support_output, y_test,
                               num_classes, k=100, T=0.1,
                               save_path=None, strategy=None, **kwargs):
    """
    DATIS Stage 1: Test Input Prioritization (official implementation of DATIS)

    Args:
        softmax_prob (np.ndarray): Softmax output of training set, with size of (num_samples, num_classes)
        train_support_output (np.ndarray): The second last layer output of training set,
                            with size of (num_samples, num_features)
        y_train (np.ndarray): Labels of training set
        test_support_output (np.ndarray)
        y_test (np.ndarray)
        num_classes (int): Num of classes
        k (int): Top-k neighbors
        T (float): Temperature parameter
    """

    'Step 1: Initialize'
    # get l2 normalized vectors
    normalizer = Normalizer(norm='l2')
    train_support_output = normalizer.transform(train_support_output)
    test_support_output = normalizer.transform(test_support_output)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_support_output, y_train)

    'Step 2: Find k neighbors in chunks'
    # dist_all, idx_all = knn.kneighbors(test_support_output, n_neighbors=k, return_distance=True)
    chunk_size = 2048
    dist_all = []
    idx_all = []

    for i in tqdm(range(0, len(test_support_output), chunk_size), desc="Find k neighbors in chunks"):
        chunk = test_support_output[i:i + chunk_size]
        dist, idx = knn.kneighbors(chunk, n_neighbors=k)

        dist_all.append(dist)
        idx_all.append(idx)
    dist_all = np.vstack(dist_all)
    idx_all = np.vstack(idx_all)

    'Step 3: Get support vectors (through training data)'
    prob_test = np.zeros((len(test_support_output), num_classes))

    for i, z in tqdm(enumerate(test_support_output),
                     total=len(test_support_output),
                     desc='Global Prioritizing'
                     ):
        # find n neighbors
        support_points = train_support_output[idx_all[i]]
        support_labels = y_train[idx_all[i]]

        # calculate denominator
        distance_sum = -np.sum((z - support_points) ** 2, axis=1) / T
        exp_distance_sum = np.exp(distance_sum)

        # original DATIS
        denominator = np.sum(exp_distance_sum)

        # calculate numerator
        for j in range(num_classes):
            num_rator = np.multiply(exp_distance_sum, (support_labels == j))
            numerator = np.sum(num_rator)
            prob_test[i][j] = numerator / denominator

    'Step 4: Get Indices'
    # get pred labels
    softmax_max_indices = np.argmax(softmax_prob, axis=1)
    # get max labels of support vectors
    max_indices = np.argmax(prob_test, axis=1)
    temp = prob_test.copy()
    for i in range(len(max_indices)):
        temp[i][max_indices[i]] = -1
    # get second max labels of support vectors
    second_max_indices = np.argmax(temp, axis=1)

    'Step 5: Get Uncertainty'
    metrics = []
    epsilon = 1e-15
    for i in range(len(max_indices)):

        # support max label is predicted label
        if (max_indices[i] == softmax_max_indices[i]):
            a = prob_test[i][second_max_indices[i]]
            b = prob_test[i][softmax_max_indices[i]]

        else:
            a = prob_test[i][max_indices[i]]
            b = prob_test[i][softmax_max_indices[i]]

        uncertainty = a / (b + epsilon)
        metrics.append(uncertainty)

    rank_lst = np.argsort(metrics)
    rank_lst = rank_lst[::-1]

    return rank_lst


def DATIS_redundancy_elimination(budget_ratio_list, rank_list, test_support_output, y_pred):
    """
    DATIS Stage 2: Redundancy Elimination
    (official implementation of DATIS, modified for new budget ratio)
    """

    'Step 1: Initialize'
    size = len(test_support_output)
    normalizer = Normalizer(norm='l2')
    test = normalizer.transform(test_support_output)

    ratio_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    pool_list = [4, 3, 3, 2, 2, 2, 2]
    weight_list = [0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]
    top_list = []
    arg_index_list = []
    for ratio_ in budget_ratio_list:
        top_list.append(int(size * ratio_))
        index = find_closest_ratio(ratio_, ratio_list)
        arg_index_list.append(index)

    'Step 2: Initialize'
    ans = []
    for i_, k in tqdm(enumerate(top_list), total=len(top_list)):
        save_point_print(f'Selecting under budget size={k}')
        index = arg_index_list[i_]
        tmp_k = int(k * pool_list[index])
        # 防止 tmp_k 超出 rank_list / 测试集长度
        tmp_k = min(tmp_k, len(rank_list))
        selected_indices = rank_list[:tmp_k]

        tmp_set = test[selected_indices, :]
        tmp_label = y_pred[selected_indices]
        kn = k
        if kn > 100:
            kn = 100
        knn = KNeighborsClassifier(n_neighbors=kn)
        knn.fit(tmp_set, tmp_label)

        chunk_size = 2048
        dist_means = []

        for start in tqdm(range(0, len(tmp_set), chunk_size), desc="Find k neighbors (tmp_set) in chunks"):
            chunk = tmp_set[start:start + chunk_size]

            dist_chunk, _ = knn.kneighbors(chunk, n_neighbors=kn, return_distance=True)

            dist_means.append(np.mean(dist_chunk, axis=1))

        distances = np.concatenate(dist_means, axis=0)

        step1_weights = np.arange(tmp_k, 0, -1)
        rank_weights = np.argsort(-distances)

        distance_weights = np.zeros(tmp_k)
        weight_k = tmp_k
        for i in rank_weights:
            distance_weights[i] = weight_k
            weight_k -= 1

        weights = (1 - weight_list[index]) * step1_weights + weight_list[index] * distance_weights
        sorted_indices = selected_indices[np.argsort(-weights)][:int(k)]
        ans.append(sorted_indices)

    return ans


