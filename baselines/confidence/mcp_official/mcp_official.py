"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "Multiple-boundary clustering and prioritization to promote
                       neural network retraining"

Note:
    This implementation is developed based on the descriptions and algorithms (e.g., pseudocode)
    provided in the original paper.

--------------------------------------------------------------------------------------
"""

from itertools import permutations
import numpy as np
from tqdm import tqdm

from baselines.confidence.simple_confidence import is_probability_vector, process_with_softmax
from utils.other_utils import save_point_print

from baselines.confidence.mcp_official.mcp_samedist import select_my_optimize

def prioritize_by_mcp_official(cand_vectors, budget_list):
    """MCP"""

    'Step 1: Setup'
    # Setup Useless params
    x_target = None
    y_test = None
    model = None

    # get num of classes
    num_classes = len(cand_vectors[0])

    'Step 2: Output Probabilities Extraction'
    # validate whether cand_vectors is composed of probability vectors
    is_probability = True
    for i, vec in enumerate(cand_vectors[:5]):
        if is_probability_vector(vec):
            print(f"cand_vectors[{i}] is a valid probability vector.")
        else:
            print(f"cand_vectors[{i}] {vec} is NOT a valid probability vector with sum {vec.sum()}.")
            is_probability = False

    if not is_probability:
        prob_vectors = process_with_softmax(cand_vectors)
    else:
        prob_vectors = cand_vectors
    prob_vectors = np.array([tensor.cpu().numpy() for tensor in prob_vectors])
    data_size = len(prob_vectors)

    # prob_loader = DataLoader(ProbVecDataset(prob_vectors), batch_size=1024, shuffle=False)

    'Step 3: Call the official implementation of MCP'
    budget_size_lst = []
    prioritized_dict = {}
    for budget in budget_list:
        if isinstance(budget, float) and 0 < budget <= 1:  # budget as a ratio
            budget_size = int(data_size * budget)
        elif isinstance(budget, int) and budget >= 1:  # budget as a specific size
            budget_size = budget
        budget_size_lst.append(budget_size)
        prioritized_dict[budget] = None

    prioritized_dict = select_my_optimize(
        model, x_target, y_test,
        selectsize=None,
        prob_vectors=prob_vectors,
        num_classes=num_classes,
        prioritized_dict=prioritized_dict,
        budget_size_lst=budget_size_lst
    )

    return prioritized_dict


