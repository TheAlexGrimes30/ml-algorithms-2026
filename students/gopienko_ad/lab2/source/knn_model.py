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
