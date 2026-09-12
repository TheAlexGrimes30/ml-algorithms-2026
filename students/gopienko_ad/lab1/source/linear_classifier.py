import numpy as np


class LinearClassifier:
    def __init__(
            self,
            l2: float = 1e-3,
            eps: float = 1e-12,
            seed: int = 42
    ):
        self.l2 = l2
        self.eps = eps
        self.seed = seed

        self.w = None
        self.b = 0.0

        self.history = {
            "objective": [],
            "recurrent_objective": [],
            "learning_rate": [],
        }

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(
            scores >= 0,
            1.0,
            -1.0
        )

    def margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return y * scores

    @staticmethod
    def margin_loss(margin: np.ndarray) -> np.ndarray:
        return (1.0 - margin) ** 2

    def one_margin(
        self,
        x: np.ndarray,
        y: float
    ) -> float:
        return float(y * (self.w @ x + self.b))

    def objective(self, X: np.ndarray, y: np.ndarray) -> float:
        margins = self.margin(X, y)
        empirical_risk = np.mean(self.margin_loss(margins))
        regularization = self.l2 * np.sum(self.w ** 2)

        return float(empirical_risk + regularization)

    def random_initialization(
            self,
            n_features: np.ndarray,
            seed: int = None
    ):
        if seed is None:
            seed = self.seed

        rng = np.random.default_rng(seed)

        self.w = rng.normal(
            loc=0.0,
            scale=0.05,
            size=n_features
        )

        self.b = 0.0

    def correlation_initialization(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean(axis=0)

        numerator = np.sum(
            X_centered * y_centered[:, None],
            axis=0
        )

        denominator = np.sqrt(
            np.sum(
                X_centered ** 2,
                axis=0
            )
            * np.sum(
                y_centered ** 2
            )
        )

        correlations = numerator / (denominator + self.eps)
        norm = np.linalg.norm(correlations)

        if norm < self.eps:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = 0.2 * correlations / norm
            self.b = float(np.mean(y))

        return correlations

    def sample_gradient(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        score = np.dot(self.w, X) + self.b
        error = score - y

        grad_w = 2.0 * error * X + 2.0 * self.l2 * self.w
        grad_b = 2.0 * error

        return grad_w, float(grad_b)

    def batch_gradient(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        scores = X @ self.w + self.b

        errors = scores - y

        grad_w = 2.0 * X.T @ errors / len(X) + 2.0 * self.l2 * self.w
        grad_b = 2.0 * np.mean(errors)

        return grad_w, float(grad_b)

    @staticmethod
    def recurrent_quality(
            previous_q: np.ndarray,
            current_loss: np.ndarray,
            alpha: float
    ) -> float:
        if previous_q is None:
            return float(current_loss)

        return float((1.0 - alpha) * previous_q + alpha * current_loss)

    def presentation_order(
            self,
            X,
            y,
            mode,
            rng
    ):

        if mode == "random":
            return rng.permutation(len(X))

        if mode == "margin":
            margins = self.margin(X, y)
            return np.argsort(np.abs(margins))

        raise ValueError(
            "mode должен быть "
            "'random' или 'margin'"
        )

    def fit_sgd_momentum(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            learning_rate: float = 0.003,
            momentum: float = 0.9,
            lr_decay: float = 2e-4,
            recurrent_alpha: float = 0.02,
            init: str = "random",
            presentation: str = "random",
            seed: int = None,
            verbose: bool = False
    ):
        if seed is None:
            seed = self.seed

        rng = np.random.default_rng(seed)

        if init == "random":
            self.random_initialization(X.shape[1], seed)

        elif init == "correlation":
            self.correlation_initialization(X, y)

        else:
            raise ValueError(
                "init должен быть 'random' или 'correlation'"
            )

        velocity_w = np.zeros_like(self.w)
        velocity_b = 0.0

        recurrent_q = None
        global_step = 0

        self.history = {
            "objective": [],
            "recurrent_objective": [],
            "accuracy": [],
            "learning_rate": [],
        }

        for epoch in range(epochs):
            indices = self.presentation_order(
                X,
                y,
                presentation,
                rng
            )

            for index in indices:
                x_i = X[index]
                y_i = y[index]

                grad_w, grad_b = self.sample_gradient(x_i, y_i)

                current_lr = (
                        learning_rate
                        / (1.0 + lr_decay * global_step)
                )

                velocity_w = (
                        momentum * velocity_w
                        + grad_w
                )

                velocity_b = (
                        momentum * velocity_b
                        + grad_b
                )

                self.w = self.w - current_lr * velocity_w
                self.b = self.b - current_lr * velocity_b

                current_margin = self.one_margin(x_i, y_i)

                current_loss = (
                        self.margin_loss(current_margin)
                        + self.l2 * np.sum(self.w ** 2)
                )

                recurrent_q = self.recurrent_quality(
                    recurrent_q,
                    current_loss,
                    recurrent_alpha
                )

                global_step += 1

            train_objective = self.objective(X, y)

            self.history["objective"].append(train_objective)
            self.history["recurrent_objective"].append(recurrent_q)
            self.history["learning_rate"].append(current_lr)

            if (
                    verbose
                    and (
                    epoch == 0
                    or (epoch + 1) % 20 == 0
                    or epoch == epochs - 1
            )
            ):
                print(
                    f"[SGD] "
                    f"epoch={epoch + 1:3d} "
                    f"Q={train_objective:.6f} "
                    f"rec_Q={recurrent_q:.6f} "
                )

        return self

    def fit_steepest_descent(
            self,
            X,
            y,
            max_iterations=300,
            tolerance=1e-9,
            init="correlation",
            seed=None,
            verbose=False
    ):
        if seed is None:
            seed = self.seed

        if init == "correlation":
            self.correlation_initialization(X, y)

        elif init == "random":
            self.random_initialization(X.shape[1], seed)

        else:
            raise ValueError(
                "init должен быть 'random' или 'correlation'"
            )

        X_aug = np.column_stack((X, np.ones(len(X))))

        theta = np.concatenate((self.w, np.array([self.b])))

        regularization_matrix = np.eye(
            X_aug.shape[1]
        )

        # Bias не регуляризуем.
        regularization_matrix[-1, -1] = 0.0

        hessian = (
                2.0 / len(X) * (X_aug.T @ X_aug)
                + 2.0 * self.l2 * regularization_matrix
        )

        self.history = {
            "objective": [],
            "recurrent_objective": [],
            "accuracy": [],
            "learning_rate": [],
        }

        recurrent_q = None

        for iteration in range(max_iterations):
            scores = X_aug @ theta
            errors = scores - y

            gradient = (
                    2.0 / len(X) * (X_aug.T @ errors)
                    + 2.0 * self.l2 * (regularization_matrix @ theta)
            )

            gradient_norm = np.linalg.norm(gradient)

            if gradient_norm < tolerance:
                break

            denominator = gradient @ hessian @ gradient


            if denominator <= self.eps:
                break

            step = (gradient @ gradient) / denominator
            theta = theta - step * gradient


            self.w = theta[:-1].copy()
            self.b = float(theta[-1])

            current_objective = self.objective(X, y)

            recurrent_q = self.recurrent_quality(
                recurrent_q,
                current_objective,
                0.1
            )

            self.history["objective"].append(
                current_objective
            )

            self.history["recurrent_objective"].append(
                recurrent_q
            )

            self.history["learning_rate"].append(
                float(step)
            )

            if (
                    verbose
                    and (
                    iteration == 0
                    or (iteration + 1) % 25 == 0
            )
            ):
                print(
                    f"[STEEPEST] "
                    f"iteration={iteration + 1:3d} "
                    f"Q={current_objective:.6f} "
                    f"step={step:.6f} "
                    f"|grad|={gradient_norm:.3e}"
                )

        return self

class RidgeClassifier:
    def __init__(self, l2: float = 1e-3):
        self.l2 = l2

        self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_aug = np.column_stack(
            (X, np.ones(len(X)))
        )

        regularization_matrix = np.eye(X_aug.shape[1])

        matrix = X_aug.T @ X_aug + len(X) * self.l2 * regularization_matrix

        right_part = (
                X_aug.T
                @ y
        )

        try:

            theta = np.linalg.solve(
                matrix,
                right_part
            )

        except np.linalg.LinAlgError:

            theta = np.linalg.pinv(matrix) @ right_part
            
        self.w = theta[
                 :-1
                 ]

        self.b = float(
            theta[
                -1
            ]
        )

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self,X: np.ndarray) -> np.ndarray:
        return np.where(
            self.decision_function(X) >= 0,
            1.0,
            -1.0
        )