
import numpy as np

from utils.dufp.density_computer.knn_kll import KnnKllDistanceComputer


class KnnPosteriorDistanceComputer(KnnKllDistanceComputer):
    """
    Bayes-inspired distance matrix for margin computation.

    We do not directly compute posterior probabilities P(y=c|z).
    Instead, we return an NLL matrix:
        D_A(z,c) = D_dens(z,c) - lambda * log(pi_c)
    """

    def __init__(
        self,
        Z_train,
        dist_type=None,
        prior_type="empirical",  # "empirical" | "uniform"
        prior_weight=1.0,  # lambda in [0, 1] (can be >1 if you want stronger prior)
        **kwargs
    ):
        super().__init__(Z_train, dist_type=dist_type, **kwargs)
        print("\ninit KnnPosteriorDistanceComputer")
        self.prior_type = prior_type
        self.prior_weight = float(prior_weight)

        self.log_pi = self._prepare_log_priors()  # (C,)

        # Fallback: parent may skip building knn_models when knn_ratio is None.
        if not hasattr(self, "knn_models") or self.knn_models is None:
            self.knn_models = self._get_knn_models_per_class()

    def _prepare_log_priors(self, eps=1e-12):
        """
        Prepare log priors log(pi_c).
        - empirical: pi_c = n_c / sum(n_c)
        - uniform: pi_c = 1/C
        """
        C = self.num_classes

        if self.prior_type == "empirical":
            pi = self.n_c.astype(float)
            pi = pi / max(pi.sum(), eps)

        elif self.prior_type == "uniform":
            pi = np.full(C, 1.0 / C, dtype=float)

        else:
            raise ValueError(
                f"Unknown prior_type: {self.prior_type} (only 'empirical' or 'uniform')"
            )

        pi = np.maximum(pi, eps)
        return np.log(pi)

    def compute(self, Z_test):
        """
        Override compute: return adjusted NLL matrix D_A when dist_type indicates posterior-like.
        """
        if "post" not in self.dist_type:
            raise ValueError("dist type is not supported")

        eps = 1e-12
        # 1) compute kth-radius matrix
        R = self._knn_kth_radius_matrix(Z_test)  # (N, C)

        # 2) compute fixed density NLL matrix
        D_dens = self._fixed_knn_kll(R, self.n_c, eps=eps)  # (N, C)

        # 3) Bayes-inspired adjustment: D_A = D_dens - lambda*log(pi)
        #    Smaller D_A means stronger evidence after prior correction.
        D_A = D_dens - (self.prior_weight * self.log_pi[None, :])
        if 'nll_posterior' in self.dist_type:
            return self.compute_nll_posterior(D_A)
        return D_A

    def compute_nll_posterior(self, D_A):
        """
        Returns log P(y=c|z), numerically stable.
        """
        print(f'compute log posterior matrix')
        log_unnorm = -D_A

        # log-sum-exp
        log_max = log_unnorm.max(axis=1, keepdims=True)
        log_sum = log_max + np.log(
            np.exp(log_unnorm - log_max).sum(axis=1, keepdims=True)
        )

        log_posterior = log_unnorm - log_sum
        return -log_posterior

    def get_prior_adjustment(self):
        """Return prior_weight * log_pi, shape (C,)."""
        return self.prior_weight * self.log_pi