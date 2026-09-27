import math

import numpy as np


def pairwise_euclidean(
        x: np.ndarray,
        y: np.ndarray
) -> np.ndarray:

    x_norm = np.sum(x ** 2, axis=1, keepdims=True)
    y_norm = np.sum(y ** 2, axis=1, keepdims=True).T

    squared = x_norm + y_norm - 2.0 * x @ y.T
    squared = np.maximum(squared, 0.0)

    return np.sqrt(squared)

def gaussian_kernel(r: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * r ** 2)

class KNNClassifier:
    def __init__(
            self,
            k: int = 5,
            eps: float = 1e-12
    ):
        self.k = k
        self.eps = eps

        self.x_train = None
        self.y_train = None
        self.classes_ = None

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> "KNNClassifier":
        self.x_train = x
        self.y_train = y

        self.x_train = x
        self.y_train = y
        self.classes_ = np.unique(y)

        return self

    def _bandwidth(self, distances: np.ndarray) -> np.ndarray:
        n_train = distances.shape[1]

        neighbor_index = min(self.k, n_train - 1)

        h = np.partition(
            distances,
            kth=neighbor_index,
            axis=1
        )[:, neighbor_index]

        zero_mask = h <= self.eps

        if np.any(zero_mask):
            positive_distances = np.where(distances > self.eps, distances, np.inf)
            fallback = np.min(positive_distances, axis=1)
            fallback = np.where( np.isfinite(fallback), fallback,1.0)
            h = np.where(zero_mask, fallback, h)

        return np.maximum(h, self.eps)

    def _class_scores_from_distances(
        self,
        distances: np.ndarray
    ) -> np.ndarray:
        h = self._bandwidth(distances)

        normalized = distances / h[:, None]
        weights = gaussian_kernel(normalized)

        scores = np.zeros(
            (
                len(distances),
                len(self.classes_),
            ),
            dtype=float,
        )

        for class_index, class_label in enumerate(self.classes_):
            class_mask = self.y_train == class_label
            scores[:, class_index,] = np.sum(weights[:, class_mask], axis=1)

        return scores

    def predict_proba(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        distances = pairwise_euclidean(x, self.x_train)
        scores = self._class_scores_from_distances(distances)
        score_sum = np.sum(scores, axis=1, keepdims=True)

        probabilities = np.divide(scores, score_sum,
            out=np.full_like(scores, 1.0 / len(self.classes_)),
            where=score_sum > self.eps,
        )

        return probabilities

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)

        if len(self.classes_) == 2:
            positive_index = int(np.flatnonzero(self.classes_ == 1)[0])\
                if np.any(self.classes_ == 1) else 1

            return probabilities[:,positive_index]

        return np.max(probabilities, axis=1)

def loo_risk_curve(
        x: np.ndarray,
        y: np.ndarray,
        k_values: np.ndarray,
        eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    valid_k = k_values[(k_values >= 1) & (k_values <= len(x) - 2)]
    distances = pairwise_euclidean(x, x)
    np.fill_diagonal(distances, np.inf)

    classes = np.unique(y)
    risks = []

    for k in valid_k:
        h = np.partition(
            distances,
            kth=k,
            axis=1,
        )[:, k]

        zero_mask = h <= eps

        if np.any(zero_mask):
            positive_distances = np.where(
                distances > eps,
                distances,
                np.inf
            )

            fallback = np.min(positive_distances, axis=1)

            fallback = np.where(
                np.isfinite(fallback),
                fallback,
                1.0
            )

            h = np.where(zero_mask, fallback, h)

        normalized = distances / np.maximum(h[:, None], eps)
        weights = gaussian_kernel(normalized)

        class_scores = np.zeros(
            (
                len(x),
                len(classes),
            ),
            dtype=float,
        )

        for class_index, class_label in enumerate(classes):
            class_scores[:, class_index] = np.sum(weights[:, y == class_label], axis=1)

        prediction = classes[np.argmax(class_scores, axis=1)]
        risk = np.mean(prediction != y)
        risks.append(float(risk))

    return valid_k, np.asarray(risks, dtype=float)

def select_best_k_loo(
    x: np.ndarray,
    y: np.ndarray,
    k_values,
) -> tuple[int, np.ndarray, np.ndarray]:
    k_values, risks = loo_risk_curve(x, y, k_values=k_values)
    best_index = int(np.argmin(risks))

    return int(k_values[best_index]), k_values, risks

def compactness_profile(
    x: np.ndarray,
    y: np.ndarray,
    max_m: int | None = None,
) -> np.ndarray:

    distances = pairwise_euclidean(x, x)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1)
    max_possible = len(x) - 1

    if max_m is None:
        max_m = max_possible
    else:
        max_m = min(int(max_m), max_possible)

    neighbor_labels = y[order[:, :max_m]]
    profile = np.mean(neighbor_labels != y[:, None], axis=0)

    return profile.astype(float)

def _ccv_weights(
    total_size: int,
    control_size: int,
) -> np.ndarray:

    L = int(total_size)
    k_control = int(control_size)
    train_size = L - k_control
    denominator = math.comb(L - 1, train_size)
    weights = []

    for m in range(1, k_control + 1):
        upper = L - 1 - m

        if upper < train_size - 1:
            weight = 0.0
        else:
            weight = math.comb(upper, train_size - 1) / denominator

        weights.append(weight)

    return np.asarray(weights, dtype=float,)


def ccv_1nn_from_profile(
    profile: np.ndarray,
    total_size: int,
    control_size: int,
) -> float:

    weights = _ccv_weights(
        total_size=total_size,
        control_size=control_size
    )
    
    length = min(len(profile), len(weights))
    return float(np.sum(profile[:length] * weights[:length]))

