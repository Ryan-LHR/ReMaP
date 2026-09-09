"""
Density estimation ablation methods for margin computation.

All classes output D_A(z, c) = -log f(z|c) - lambda * log(pi_c),
consistent with KnnPosteriorDistanceComputer, so that the downstream
margin computation remains:
    margin = D_A(z, c_pred) - D_A(z, c_competitor)
           = log [ P(c_competitor|z) / P(c_pred|z) ]

Methods implemented:
    - KdeDensityComputer:       Kernel Density Estimation (Gaussian kernel)
    - GaussianDensityComputer:  Single Gaussian per class (Ledoit-Wolf shrinkage)
    - GmmDensityComputer:       Gaussian Mixture Model per class
    - VmfDensityComputer:       Von Mises-Fisher distribution per class
"""

import math
import numpy as np
from tqdm import tqdm

from utils.dufp.dufp_utils import try_get_param


class BaseDensityComputer:
    """
    Base class for density estimation ablation.
    Handles: label mapping, class counts, prior computation, and
    the unified compute() interface.
    """

    def __init__(self, Z_train, num_classes=None, dist_type=None,
                 prior_type="empirical", prior_weight=1.0, **kwargs):
        self.Z_train = Z_train
        self.num_classes = num_classes
        self.dist_type = dist_type if dist_type is not None else ""
        self.prior_type = prior_type
        self.prior_weight = float(prior_weight)
        self.kwargs = kwargs

        self.dim = Z_train.shape[1]
        self.truths_train = kwargs.get("truths_train", None)

        self.classes = kwargs.get("classes", None)
        if self.classes is None:
            self.classes = np.unique(self.truths_train)
        self.classes = np.asarray(self.classes)

        self.idx_train = self._labels_to_row_index(self.truths_train, self.classes)
        self.n_c = np.bincount(self.idx_train, minlength=self.num_classes).astype(int)

        self.log_pi = self._prepare_log_priors()

    def _labels_to_row_index(self, truths_train, classes):
        if classes.size > 1 and np.all(classes[:-1] <= classes[1:]):
            return np.searchsorted(classes, truths_train)
        class_to_row = {c: i for i, c in enumerate(classes)}
        return np.array([class_to_row[t] for t in truths_train], dtype=int)

    def _prepare_log_priors(self, eps=1e-12):
        C = self.num_classes
        if self.prior_type == "empirical":
            pi = self.n_c.astype(float)
            pi = pi / max(pi.sum(), eps)
        elif self.prior_type == "uniform":
            pi = np.full(C, 1.0 / C, dtype=float)
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}")
        pi = np.maximum(pi, eps)
        return np.log(pi)

    def compute(self, Z_test):
        """
        Returns D_A (N_test, C):
            D_A[i, c] = -log f(z_i | c)  -  lambda * log(pi_c)
        """
        D_raw = self._compute_raw_nll(Z_test)  # (N, C)
        D_A = D_raw - (self.prior_weight * self.log_pi[None, :])
        return D_A

    def _compute_raw_nll(self, Z_test):
        """
        Subclass must override.
        Returns raw NLL matrix (N_test, C):  -log f(z|c)
        """
        raise NotImplementedError

    def get_prior_adjustment(self):
        """Return prior_weight * log_pi, shape (C,).
        D_A = D_dens - adjustment, so D_dens = D_A + adjustment."""
        return self.prior_weight * self.log_pi


class KdeDensityComputer(BaseDensityComputer):
    """
    Per-class Kernel Density Estimation using Gaussian kernel.
    Bandwidth: Silverman's rule h = n_c^{-1/(d+4)}.
    """

    def __init__(self, Z_train, num_classes=None, dist_type=None, **kwargs):
        super().__init__(Z_train, num_classes=num_classes, dist_type=dist_type, **kwargs)
        print("\ninit KdeDensityComputer")

        # parse optional bandwidth from dist_type, e.g. "kde_bw0.1"
        self.bandwidth = try_get_param(self.dist_type, "bw", float, None)

        self.kde_models = self._fit_kde_models()

    def _fit_kde_models(self):
        from sklearn.neighbors import KernelDensity

        models = [None] * self.num_classes
        for c in tqdm(range(self.num_classes), desc="Fitting per-class KDE"):
            Z_c = self.Z_train[self.idx_train == c]
            n_c = Z_c.shape[0]

            # Silverman's rule of thumb for high-dim
            if self.bandwidth is not None:
                bw = self.bandwidth
            else:
                bw = n_c ** (-1.0 / (self.dim + 4))

            kde = KernelDensity(kernel='gaussian', bandwidth=bw, metric='euclidean')
            kde.fit(Z_c)
            models[c] = kde

        return models

    def _compute_raw_nll(self, Z_test):
        N = Z_test.shape[0]
        D = np.zeros((N, self.num_classes), dtype=float)

        for c in tqdm(range(self.num_classes), desc="KDE scoring"):
            # score_samples returns log f(z|c)
            log_density = self.kde_models[c].score_samples(Z_test)
            D[:, c] = -log_density  # NLL

        return D

class GaussianDensityComputer(BaseDensityComputer):
    """
    Per-class single Gaussian density with Ledoit-Wolf shrinkage covariance.
    NLL = 0.5*(z-mu)^T Sigma^{-1} (z-mu) + 0.5*log|Sigma| + (d/2)*log(2*pi)
    """

    def __init__(self, Z_train, num_classes=None, dist_type=None, **kwargs):
        super().__init__(Z_train, num_classes=num_classes, dist_type=dist_type, **kwargs)
        print("\ninit GaussianDensityComputer")

        # parse covariance type: "full" (Ledoit-Wolf) or "diag"
        cov_type_str = try_get_param(self.dist_type, "covtype", str, "diag")
        self.cov_type = cov_type_str  # "full" or "diag"

        self.means = [None] * self.num_classes       # (d,)
        self.precisions = [None] * self.num_classes   # (d, d) or (d,)
        self.log_det = [None] * self.num_classes      # scalar

        self._fit_gaussians()

    def _fit_gaussians(self):
        from sklearn.covariance import LedoitWolf

        for c in tqdm(range(self.num_classes), desc="Fitting per-class Gaussian"):
            Z_c = self.Z_train[self.idx_train == c]
            mu_c = Z_c.mean(axis=0)
            self.means[c] = mu_c

            if self.cov_type == "diag":
                # diagonal covariance
                var_c = Z_c.var(axis=0) + 1e-6  # (d,)
                self.precisions[c] = 1.0 / var_c  # (d,)
                self.log_det[c] = np.sum(np.log(var_c))  # log|Sigma|

            else:
                # full covariance with Ledoit-Wolf shrinkage
                try:
                    lw = LedoitWolf()
                    lw.fit(Z_c)
                    cov_c = lw.covariance_
                    prec_c = lw.precision_
                    sign, log_det_c = np.linalg.slogdet(cov_c)
                    if sign <= 0:
                        raise np.linalg.LinAlgError("Non-positive definite")
                    self.precisions[c] = prec_c
                    self.log_det[c] = log_det_c
                except Exception:
                    # fallback to diagonal
                    print(f"  [Gaussian] class {c}: fallback to diag covariance")
                    var_c = Z_c.var(axis=0) + 1e-6
                    self.precisions[c] = np.diag(1.0 / var_c)
                    self.log_det[c] = np.sum(np.log(var_c))

    def _compute_raw_nll(self, Z_test):
        N = Z_test.shape[0]
        D = np.zeros((N, self.num_classes), dtype=float)
        const = 0.5 * self.dim * np.log(2.0 * np.pi)

        for c in range(self.num_classes):
            diff = Z_test - self.means[c][None, :]  # (N, d)

            if self.cov_type == "diag" and self.precisions[c].ndim == 1:
                # diagonal: (z-mu)^T diag(1/var) (z-mu) = sum( diff^2 * prec )
                maha = np.sum(diff ** 2 * self.precisions[c][None, :], axis=1)
            else:
                # full: (z-mu)^T Sigma^{-1} (z-mu)
                maha = np.sum(diff @ self.precisions[c] * diff, axis=1)

            D[:, c] = 0.5 * maha + 0.5 * self.log_det[c] + const

        return D



class GmmDensityComputer(BaseDensityComputer):
    """
    Per-class Gaussian Mixture Model.
    Default: n_components=3, covariance_type='diag' (robust in high-dim).
    """

    def __init__(self, Z_train, num_classes=None, dist_type=None, **kwargs):
        super().__init__(Z_train, num_classes=num_classes, dist_type=dist_type, **kwargs)
        print("\ninit GmmDensityComputer")

        # parse n_components from dist_type, e.g. "gmm_ncomp5"
        self.n_components = try_get_param(self.dist_type, "ncomp", int, 5)
        # parse covariance_type: "diag" (default) or "full"
        self.gmm_cov_type = try_get_param(self.dist_type, "covtype", str, "diag")

        self.gmm_models = self._fit_gmm_models()

    def _fit_gmm_models(self):
        from sklearn.mixture import GaussianMixture

        models = [None] * self.num_classes
        for c in tqdm(range(self.num_classes), desc="Fitting per-class GMM"):
            Z_c = self.Z_train[self.idx_train == c]
            n_c = Z_c.shape[0]

            # ensure n_components <= n_samples
            n_comp = min(self.n_components, n_c)

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.gmm_cov_type,
                max_iter=50,
                n_init=3,
                random_state=42,
            )
            gmm.fit(Z_c)
            models[c] = gmm

        return models

    def _compute_raw_nll(self, Z_test):
        N = Z_test.shape[0]
        D = np.zeros((N, self.num_classes), dtype=float)

        for c in tqdm(range(self.num_classes), desc="GMM scoring"):
            # score_samples returns log f(z|c)
            log_density = self.gmm_models[c].score_samples(Z_test)
            D[:, c] = -log_density

        return D


class VmfDensityComputer(BaseDensityComputer):
    """
    Per-class Von Mises-Fisher density.
    Assumes input features are already L2-normalized (on unit hypersphere).

    PDF:  f(z|c) = C_d(kappa_c) * exp(kappa_c * mu_c^T z)
    NLL: -log f(z|c) = -kappa_c * mu_c^T z  -  log C_d(kappa_c)

    where log C_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2*pi) - log I_{d/2-1}(kappa)
    and I_v is the modified Bessel function of the first kind.

    kappa estimation: Banerjee et al. (2005) MLE approximation
        kappa_c ≈ R_bar * (d - R_bar^2) / (1 - R_bar^2)
    where R_bar = ||mean(z_c)||.
    """

    def __init__(self, Z_train, num_classes=None, dist_type=None, **kwargs):
        super().__init__(Z_train, num_classes=num_classes, dist_type=dist_type, **kwargs)
        print("\ninit VmfDensityComputer")

        self.mus = [None] * self.num_classes       # unit mean direction (d,)
        self.kappas = [None] * self.num_classes     # concentration scalar
        self.log_C_d = [None] * self.num_classes    # log normalizing constant

        self._fit_vmf()

    def _fit_vmf(self):
        for c in tqdm(range(self.num_classes), desc="Fitting per-class vMF"):
            Z_c = self.Z_train[self.idx_train == c]

            # mean direction
            mean_vec = Z_c.mean(axis=0)  # (d,)
            R_bar = np.linalg.norm(mean_vec)
            R_bar = np.clip(R_bar, 1e-10, 1.0 - 1e-7)  # avoid boundary

            mu_c = mean_vec / np.linalg.norm(mean_vec)
            self.mus[c] = mu_c

            # kappa MLE approximation (Banerjee et al. 2005)
            d = self.dim
            kappa_c = R_bar * (d - R_bar ** 2) / (1.0 - R_bar ** 2)
            kappa_c = np.clip(kappa_c, 1e-3, 1e5)  # numerical safety
            self.kappas[c] = kappa_c

            # log normalizing constant
            self.log_C_d[c] = self._compute_log_Cd(d, kappa_c)

        kappa_arr = [self.kappas[c] for c in range(self.num_classes)]
        print(f"  vMF kappa range: [{min(kappa_arr):.2f}, {max(kappa_arr):.2f}]")

    @staticmethod
    def _compute_log_Cd(d, kappa):
        """
        log C_d(kappa) = (d/2 - 1)*log(kappa) - (d/2)*log(2*pi) - log I_{d/2-1}(kappa)

        Use scipy.special.ive for numerically stable Bessel:
            ive(v, kappa) = I_v(kappa) * exp(-kappa)
            => log I_v(kappa) = log(ive(v, kappa)) + kappa
        """
        from scipy.special import ive

        v = d / 2.0 - 1.0
        log_ive = np.log(np.maximum(ive(v, kappa), 1e-300))
        log_bessel = log_ive + kappa  # log I_v(kappa)

        log_Cd = v * np.log(np.maximum(kappa, 1e-300)) \
                 - (d / 2.0) * np.log(2.0 * np.pi) \
                 - log_bessel

        return log_Cd

    def _compute_raw_nll(self, Z_test):
        N = Z_test.shape[0]
        D = np.zeros((N, self.num_classes), dtype=float)

        for c in range(self.num_classes):
            # mu_c^T z  for all test points
            cos_sim = Z_test @ self.mus[c]  # (N,)
            # NLL = -kappa * mu^T z - log C_d(kappa)
            D[:, c] = -self.kappas[c] * cos_sim - self.log_C_d[c]

        return D


class FixedRadiusDensityComputer(BaseDensityComputer):
    """
    Fixed-radius (Parzen window / uniform kernel) density estimation.
    Dual of KNN density: KNN fixes k and lets r adapt; this fixes r and lets count adapt.

    Per-class density:
        f(z|c) = count_r(z, c) / (n_c * V_d * r^d)

    NLL:
        -log f(z|c) = -log(count_r(z,c)) + log(n_c) + log(V_d) + d*log(r)

    For margin computation, log(V_d) + d*log(r) is constant across classes and cancels.

    Radius selection (default):
        For each class c, build a temporary KNN (k=small), compute median of
        k-th neighbor distances across all training points, then take the
        global median as the fixed radius r. This gives a data-driven,
        principled default without manual tuning.
    """

    def __init__(self, Z_train, num_classes=None, dist_type=None, **kwargs):
        super().__init__(Z_train, num_classes=num_classes, dist_type=dist_type, **kwargs)
        print("\ninit FixedRadiusDensityComputer")

        self.knn_metric = "cosine"  # consistent with KNN density code

        # parse optional radius from dist_type, e.g. "fixedradius_r0.5"
        self.radius = try_get_param(self.dist_type, "r", float, None)

        if self.radius is None:
            self.radius = self._auto_select_radius()
            print(f"  Auto-selected radius (in euclidean space): {self.radius:.6f}")
        else:
            print(f"  User-specified radius: {self.radius:.6f}")

        # convert euclidean radius to cosine-metric radius for sklearn
        # sklearn cosine distance = 1 - cos(theta)
        # euclidean distance after L2-norm = sqrt(2 * cosine_distance)
        # so cosine_distance = euclidean^2 / 2
        self.radius_cosine = (self.radius ** 2) / 2.0
        print(f"  Corresponding cosine-metric radius: {self.radius_cosine:.6f}")

        # build per-class NearestNeighbors for radius query
        self.nn_models = self._build_nn_models()

        # precompute log(V_d) + d*log(r) (constant, cancels in margin)
        self.log_Vd = self._get_log_V_d(self.dim)
        self.const_term = self.log_Vd + self.dim * np.log(max(self.radius, 1e-30))

    @staticmethod
    def _get_log_V_d(d):
        return 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)

    def _auto_select_radius(self):
        """
        Auto-select radius from training data:
        For each class, compute 10-th nearest neighbor distances for all
        training points in that class, then take the global median.
        This gives a scale-aware, data-driven radius.
        """
        from sklearn.neighbors import NearestNeighbors

        k_probe = 10  # small k for probing scale
        all_kth_dists = []

        for c in range(self.num_classes):
            Z_c = self.Z_train[self.idx_train == c]
            n_c = Z_c.shape[0]
            k_use = min(k_probe, n_c - 1)
            if k_use < 1:
                continue

            nn = NearestNeighbors(n_neighbors=k_use, metric=self.knn_metric, algorithm="auto")
            nn.fit(Z_c)
            dists, _ = nn.kneighbors(Z_c, n_neighbors=k_use, return_distance=True)
            kth_dists = dists[:, -1]  # k-th neighbor distance (in cosine metric)

            # convert cosine metric to euclidean
            kth_dists_eucl = np.sqrt(np.maximum(0.0, 2.0 * kth_dists))
            all_kth_dists.append(kth_dists_eucl)

        all_kth_dists = np.concatenate(all_kth_dists)
        radius = np.median(all_kth_dists)
        return max(radius, 1e-6)

    def _build_nn_models(self):
        """
        Build per-class NearestNeighbors models for radius_neighbors queries.
        We set n_neighbors=1 as placeholder; actual queries use radius_neighbors.
        """
        from sklearn.neighbors import NearestNeighbors

        models = [None] * self.num_classes
        for c in tqdm(range(self.num_classes), desc="Building per-class NN (fixed-radius)"):
            Z_c = self.Z_train[self.idx_train == c]
            nn = NearestNeighbors(metric=self.knn_metric, algorithm="auto")
            nn.fit(Z_c)
            models[c] = nn
        return models

    def _compute_raw_nll(self, Z_test):
        """
        NLL = -log(count_r(z,c)) + log(n_c) + log(V_d) + d*log(r)

        When count_r = 0, set count to a small floor (0.5) to avoid -log(0).
        """
        N = Z_test.shape[0]
        D = np.zeros((N, self.num_classes), dtype=float)

        for c in tqdm(range(self.num_classes), desc="Fixed-radius counting"):
            nn = self.nn_models[c]
            # radius_neighbors returns list of arrays (variable length per query)
            neighbors_list = nn.radius_neighbors(
                Z_test, radius=self.radius_cosine, return_distance=False
            )
            # count neighbors for each test point
            counts = np.array([len(nbrs) for nbrs in neighbors_list], dtype=float)

            # floor: avoid log(0); 0.5 is a standard continuity correction
            counts = np.maximum(counts, 0.5)

            D[:, c] = (
                -np.log(counts)
                + np.log(self.n_c[c] + 1e-12)
                + self.const_term  # log(V_d) + d*log(r), constant across classes
            )

        return D