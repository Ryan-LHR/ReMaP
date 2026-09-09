"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "CertPri: Certifiable Prioritization for Deep Neural Networks
                       via Movement Cost in Feature Space"

Note:
    This code is adapted from the official implementation provided by the authors.
    Minor modifications may have been made for compatibility or integration purposes.
--------------------------------------------------------------------------------------
"""

import numpy as np
from tqdm import tqdm

from baselines.confidence.certpri.model_boundaryPriCenter import inverper_c
from utils.data_utils import load_dataset_to_ndarray

def prioritize_by_certpri(model, cand_set, cand_vectors, seed, device):
    """
    CertPri (official implementation of CertPri, modified for torch framework)
    """

    'Step 1: Setup'
    R_LI = 0.05
    cand_x = load_dataset_to_ndarray(cand_set)
    total_sample_num = len(cand_vectors)
    np.random.seed(seed)

    'Step 2: Call the official implementation of CertPri'
    score = []
    for i in tqdm(range(total_sample_num), desc='inverper_c: '):
        pred_i = cand_vectors[i].cpu().numpy()
        pred_i = np.expand_dims(pred_i, axis=0)
        x = cand_x[i]
        # x = np.expand_dims(x, axis=1)
        res_tmp = inverper_c(model, x, 3, 5, R_LI, norm=np.inf, pool_factor=3, y_pred=pred_i, device=device)
        score.append(res_tmp)

    'Step 3: Prioritize'
    indexs = np.argsort(score)
    prioritized_indices = indexs.tolist()

    return prioritized_indices