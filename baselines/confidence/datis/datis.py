"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "Distance-Aware Test Input Selection for Deep Neural Networks"

Note:
    This code is adapted from the official implementation provided by the authors.
    Minor modifications may have been made for compatibility or integration purposes.
--------------------------------------------------------------------------------------
"""
import torch

from utils.dufp.dufp_utils import convert_vectors_to_probs, get_target_layer_name, get_labels_and_classes
from utils.other_utils import save_point_print, load_features
from utils.model_utils import extract_layer_output, flatten_layer_output
from baselines.confidence.datis.datis_utils import *

def prioritize_by_datis(model, model_name, train_loader, cand_loader,
                        cand_vectors, cand_labels, budget_list, args,
                        train_correct, train_vectors):
    """Selection by DATIS"""

    'Step 1: Output Probabilities Extraction'

    # get true labels
    y_train, y_cand = get_labels_and_classes(
        args,
        model_name,
        train_loader,
        cand_loader,
    )

    if not isinstance(y_train, np.ndarray):
        y_train = np.asarray(y_train)
        y_cand = np.asarray(y_cand)

    # get labels
    y_pred_cand = np.array([t.cpu() for t in cand_labels])  # pred labels

    # get num of classes
    classes = np.unique(y_train)
    num_classes = classes.size

    # get softmax outputs of candidate set
    # softmax_cand_prob = np.array([tensor.cpu().numpy() for tensor in cand_vectors])
    softmax_cand_prob = convert_vectors_to_probs(cand_vectors)

    # get outputs of the second last layer (same as datis/dimp)
    layer_name = get_target_layer_name(model_name)

    if not args.load_feature:
        # extract
        train_support_output = extract_layer_output(model, train_loader, layer_name)
        cand_support_output = extract_layer_output(model, cand_loader, layer_name)
        # flatten layer outputs
        train_support_output = flatten_layer_output(train_support_output)
        cand_support_output = flatten_layer_output(cand_support_output)

    else:
        train_kwargs = {'model': model, 'loader': train_loader}
        cand_kwargs = {'model': model, 'loader': cand_loader}

        train_support_output = load_features(args, layer_name, label='Train', **train_kwargs)
        cand_support_output = load_features(args, layer_name, label=f'Candidate_{args.cand_type}', **cand_kwargs)

    'Step 2: Global Prioritization / Selection'
    rank_lst = DATIS_test_input_selection(softmax_cand_prob, train_support_output, y_train,
                                          cand_support_output, y_pred_cand, num_classes)

    if args.method == 'datis_no_selection':
        prioritized_indices = rank_lst.tolist()
        return prioritized_indices

    'Step 3: Redundancy Elimination Under Different Budget'
    selected_results = DATIS_redundancy_elimination(budget_list, rank_lst, cand_support_output, y_pred_cand)

    prioritized_dict = {}
    for i, budget in enumerate(budget_list):
        prioritized_dict[budget] = selected_results[i].tolist()

    return prioritized_dict

