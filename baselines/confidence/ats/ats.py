"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "Adaptive test selection for deep neural networks"

Note:
    This code is adapted from the official implementation provided by the authors.
    Minor modifications may have been made for compatibility or integration purposes.
--------------------------------------------------------------------------------------
"""

import numpy as np

from baselines.confidence.ats.measure.ATSmeasure import ATSmeasure
from baselines.confidence.ats.selection.PrioritySelectStrategy import PrioritySelectStrategy
from utils.dufp.dufp_utils import get_labels_and_classes


class ATS(object):
    def __init__(self):
        self.ats_method = PrioritySelectStrategy()
        self.ats_measure = ATSmeasure()

    def get_priority_sequence(self, Tx, Ty, n, M, base_path=None, prefix=None, is_save_ps=False,
                             th=0.001, cand_prob=None):
        return self.ats_method.get_priority_sequence(Tx, Ty, n, M, base_path=base_path,  prefix=prefix,
                                                     is_save_ps=is_save_ps, th=th, cand_prob=cand_prob)


def prioritize_by_ats(train_loader, cand_loader, cand_vectors, cand_labels, args, model_name):
    """
    ATS (Official implementation of ATS)
    """

    'Step 1: Setup'
    # get true labels
    _, y_cand = get_labels_and_classes(
        args,
        model_name,
        train_loader,
        cand_loader,
    )
    num_classes = len(set(y_cand))

    y_sel_psedu = np.array([t.cpu() for t in cand_labels])  # pred labels
    x_sel = None
    ori_model = None

    ats = ATS()
    # get output probabilities of candidate set
    cand_prob = np.array([tensor.cpu().numpy() for tensor in cand_vectors])

    'Step 2: Call the official implementation of ATS'
    div_rank, _, _ = ats.get_priority_sequence(x_sel, y_sel_psedu, num_classes, ori_model,
                                               th=0.001, cand_prob=cand_prob)
    prioritized_indices = div_rank.tolist()

    return prioritized_indices