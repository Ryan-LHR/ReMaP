from baselines.confidence.ats import ats_config


class ClusterTestStep(object):
    def split_data_region_with_idx(self, Tx_prob_matrixc, i, idx):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]

        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]
        idx_1 = idx[Tx_i_prob_vec < ats_config.boundary]
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]
        idx_0 = idx[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.up_boundary)]
        idx_2 = idx[(Tx_i_prob_vec >= ats_config.up_boundary)]

        # --- minimal fix: ensure S0_i / idx_0 is non-empty and remove duplicates ---
        if len(idx_0) == 0:
            import numpy as np
            b = ats_config.boundary
            ub = ats_config.up_boundary

            # distance to interval [b, ub): if x<b -> b-x; if x>=ub -> x-ub; inside -> 0
            dist = np.where(Tx_i_prob_vec < b, b - Tx_i_prob_vec,
                            np.where(Tx_i_prob_vec >= ub, Tx_i_prob_vec - ub, 0.0))
            k = int(np.argmin(dist))

            # add the nearest sample into S0
            S0_i = Tx_prob_matrixc[k:k + 1]
            idx_0 = idx[k:k + 1]

            # remove duplicated sample from S1 or S2 (based on original region)
            if Tx_i_prob_vec[k] < b:
                mask_keep = idx_1 != idx[k]
                S1_i = S1_i[mask_keep]
                idx_1 = idx_1[mask_keep]
            else:
                mask_keep = idx_2 != idx[k]
                S2_i = S2_i[mask_keep]
                idx_2 = idx_2[mask_keep]
        # --- end minimal fix ---

        return S2_i, idx_2, S0_i, idx_0, S1_i, idx_1

    def split_data_region(self, Tx_prob_matrixc, i):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]

        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        return S2_i, S0_i, S1_i
