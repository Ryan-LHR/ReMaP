import math

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from utils.dufp.dufp_utils import try_get_param


class KnnKllDistanceComputer:
    """
    Compute class-wise distance/density matrix of test features.
    """

    def __init__(self, Z_train, num_classes=None, dist_type='euclidean', **kwargs):
        """
        Initialize the computer with training features.

        Args:
            Z_train (np.ndarray): Training features, shape (N_train, D).
            dist_type (str): Distance type string.
        """
        print("\ninit KnnKllDistanceComputer")

        self.Z_train = Z_train
        self.dist_type = dist_type
        self.kwargs = kwargs

        self.num_classes = num_classes
        self.dim = Z_train.shape[1]

        self.truths_train = self.kwargs.get("truths_train", None)
        self.alpha = self.kwargs.get("alpha", None)

        self._further_init()

        self._prepare_knn_params()

        print(f"Current params: dist type is:{self.dist_type}, "
              f"knn_k is:{self.knn_k}, "
              f"knn_ratio is:{self.knn_ratio}, "
              f"knn_metric is:{self.knn_metric}, "
              f"k_per_class is:{self.k_per_class}, "
              f"k_dict is:{self.knn_k_dict}"
              )

    def _get_log_V_d(self, d):
        """
        Compute log(V_d) where V_d is the volume of the unit d-dimensional Euclidean ball:
            V_d = pi^(d/2) / Gamma(d/2 + 1)
        """
        # log(V_d) = (d/2)*log(pi) - log(Gamma(d/2 + 1))
        return 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)

    def _further_init(self):
        """
        Further complex init
        """
        self.classes = self.kwargs.get("classes", None)
        if self.classes is None:
            self.classes = np.unique(self.truths_train)
        self.classes = np.asarray(self.classes)

        self.idx_train = self._labels_to_row_index(self.truths_train, self.classes)

        # get data_size of each class
        self.n_c = np.bincount(self.idx_train, minlength=self.num_classes).astype(int)

        # get volume constant (log)
        self.log_Vd = self._get_log_V_d(self.dim)

    def _labels_to_row_index(self, truths_train, classes):
        """
        Map original label values to [0..C-1] row index aligned with class labels.
        """
        if classes.size > 1 and np.all(classes[:-1] <= classes[1:]):
            return np.searchsorted(classes, truths_train)
        class_to_row = {c: i for i, c in enumerate(classes)}
        return np.array([class_to_row[t] for t in truths_train], dtype=int)

    def _prepare_knn_params(self):
        """
        Prepare kNN parameters.
        """
        # default value
        self.knn_k = 20
        self.knn_ratio = None
        self.k_per_class = True
        # self.alpha = None

        self.knn_k = try_get_param(self.dist_type, "k", int, self.knn_k)
        self.knn_ratio = try_get_param(self.dist_type, "kratio", float, self.knn_ratio)
        self.k_per_class = try_get_param(self.dist_type, "kperclass", bool, self.k_per_class)
        self.alpha = try_get_param(self.dist_type, "alpha", float, self.alpha)

        if self.knn_ratio == None and self.alpha is not None:
            avg_class_num = self.Z_train.shape[0] / self.num_classes
            self.knn_ratio = self.alpha / (avg_class_num ** 0.5)

        self.knn_metric = "cosine"  # knn distance metric
        self.knn_k_dict = None

        if self.knn_ratio is None:  # constant knn_k
            return

        if self.k_per_class:
            self._compute_knn_k_dict()

        else:
            train_size = self.Z_train.shape[0]
            train_class_size = train_size / self.num_classes
            self.knn_k = int(round(train_class_size * self.knn_ratio))

        self.knn_models = self._get_knn_models_per_class()

    def _compute_knn_k_dict(self, ):
        """
        Compute the k for each class
        """
        self.knn_k_dict = {}
        for c in range(self.num_classes):
            k_c = int(round(self.n_c[c] * self.knn_ratio))
            k_c = max(1, k_c)
            k_c = min(k_c, self.n_c[c])
            self.knn_k_dict[c] = k_c

        return

    def _get_knn_models_per_class(self):
        """
        Build per-class NearestNeighbors models.
        Optional reuse via kwargs['knn_models'].
        """
        knn_models = [None] * self.num_classes

        for c in tqdm(range(self.num_classes), desc="Building per-class knn models"):
            Z_train_c = self.Z_train[self.idx_train == c]
            k_c = self._get_k_c(c)
            if Z_train_c.shape[0] < k_c or k_c <= 0:
                raise ValueError("Data size and k_c is not match")

            nn = NearestNeighbors(n_neighbors=k_c, metric=self.knn_metric, algorithm="auto")
            nn.fit(Z_train_c)
            knn_models[c] = nn

        return knn_models

    def _get_k_c(self, c):
        return self.knn_k_dict.get(c, self.knn_k) if self.knn_k_dict is not None else self.knn_k

    def _knn_kth_radius_matrix(self, Z_test):
        """
        Compute r_k(z,c) for all test z and class c:
          r_k(z,c) = distance from z to k-th nearest neighbor in class c train set.
        Return: R (N_test, C)
        """
        N = Z_test.shape[0]
        R = np.full((N, self.num_classes), np.inf, dtype=float)

        for c in tqdm(range(self.num_classes), desc="compute kth radius: "):
            nn = self.knn_models[c]
            k_c = self._get_k_c(c)

            dist_knn, _ = nn.kneighbors(Z_test, n_neighbors=k_c, return_distance=True)
            kth = dist_knn[:, -1]

            if self.knn_metric == "cosine":
                R[:, c] = np.sqrt(np.maximum(0.0, 2.0 * kth))
            elif self.knn_metric == "euclidean":
                R[:, c] = kth
            else:
                R[:, c] = kth

        R = np.maximum(R, 1e-6)
        return R

    def compute(self, Z_test):
        """
        Compute distance matrix for the given test features.

        Args:
            Z_test (np.ndarray): Test features, shape (N_test, D).

        Returns:
            np.ndarray: Distance matrix, shape (N_test, C).
        """

        if "knn_nll" in self.dist_type:
            eps = 1e-12
            R = self._knn_kth_radius_matrix(Z_test)
            if "fixed" in self.dist_type:
                D = self._fixed_knn_kll(R, self.n_c, eps=eps)  # fixed
            else:
                D = self.dim * np.log(R + eps) + np.log(self.n_c[None, :] + eps)  # origin
            return D
        raise ValueError(f"Unknown dist_type: {self.dist_type}")

    def _fixed_knn_kll(self, R, n_c, eps=1e-12):
        """
        Fixed kNN-KLL (NLL) distance:
            D = d*log(R) + log(n_c) - log(k_c)

        where k_c is per-class k if available (kperclass=True), otherwise a fixed knn_k.
        """
        print(f"fixed_knn_kll")
        if self.knn_k_dict is not None:
            k_vec = np.array(
                [self.knn_k_dict.get(c, self.knn_k) for c in range(self.num_classes)],
                dtype=float
            )
        else:
            k_vec = np.full(self.num_classes, float(self.knn_k), dtype=float)

        k_vec = np.maximum(k_vec, 1.0)  # avoid log(0)
        # k_eff = np.maximum(k_vec - 1, 1)  # avoid log(0)

        D = (
                self.dim * np.log(R + eps)
                + np.log(n_c[None, :] + eps)
                - np.log(k_vec[None, :] + eps)
                # - np.log(k_eff[None, :] + eps)
                # + self.log_Vd  # volume constant (not necessary)
        )
        return D