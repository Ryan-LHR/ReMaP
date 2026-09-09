"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "Robust Test Selection for Deep Neural Networks"

Note:
    This code is adapted from the official implementation provided by the authors.
    Minor modifications may have been made for compatibility or integration purposes.
--------------------------------------------------------------------------------------
"""
import numpy as np

from baselines.confidence.rts.selection_method.selection_utils import prepare_rank_ps
from utils.data_utils import load_dataset_to_ndarray

def prioritize_by_rts(dataset_name, train_set, cand_set, train_vectors, train_labels, cand_vectors, seed):
    """
    RTS (official implementation of RTS, modified for torch dataloader and model framework)
    """

    'Step 1: Setup'
    # Convert inputs to ndarray
    x_train = load_dataset_to_ndarray(train_set)

    # y_train = np.array(train_labels)
    y_train = np.array([t.cpu() for t in train_labels])

    x_select = load_dataset_to_ndarray(cand_set)

    if x_train.ndim == 4:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_select = x_select.transpose(0, 2, 3, 1)

    data_name = dataset_name
    nb_classes = len(set([label for _, label in train_set]))

    rank_name_list = ["DeepDAC"]

    # useless params in RTS
    model_name = None
    model_path = None
    cov_initer = None
    max_select_size = None
    save_path = None

    # get output probabilities
    train_prob = np.array([tensor.cpu().numpy() for tensor in train_vectors])
    cand_prob = np.array([tensor.cpu().numpy() for tensor in cand_vectors])

    # Small batch test
    # small_batch_test = True
    small_batch_test = False
    # if small_batch_test:
    #     x_train = x_train[:6000]  # for test
    #     y_train = y_train[:6000]  # for test
    #     x_select = x_select[:2000]  # for test
    #     train_prob = train_prob[:6000]  # for test
    #     cand_prob = cand_prob[:2000]  # for test

    'Step 2: Call the official implementation of RTS'
    rank_lst = prepare_rank_ps(x_train, y_train, rank_name_list, model_path, x_select, save_path, cov_initer,
                            max_select_size, nb_classes, data_name, model_name,
                            train_pro=train_prob, select_pro=cand_prob, seed=seed)

    'Step 3: Prioritize'
    prioritized_indices = rank_lst.tolist()
    return prioritized_indices