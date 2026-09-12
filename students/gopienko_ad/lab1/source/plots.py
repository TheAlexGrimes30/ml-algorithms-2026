import matplotlib.pyplot as plt
import numpy as np

from students.gopienko_ad.lab1.source.metrics import confusion_matrix


def plot_feature_correlations(
    feature_names,
    correlations
):
    order = np.argsort(
        np.abs(
            correlations
        )
    )

    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.barh(
        feature_names[
            order
        ],
        correlations[
            order
        ]
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Correlation with target"
    )

    plt.title(
        "Корреляция признаков с классом"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "01_correlations.png",
        dpi=160
    )

    plt.close()


def plot_margins(
    model,
    X,
    y
):
    margins = model.margin(
        X,
        y
    )

    correct = (
        margins > 0
    )

    incorrect = (
        margins <= 0
    )

    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.hist(
        margins[
            correct
        ],
        bins=25,
        alpha=0.7,
        label="M > 0"
    )

    if np.any(
        incorrect
    ):

        plt.hist(
            margins[
                incorrect
            ],
            bins=15,
            alpha=0.7,
            label="M <= 0"
        )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2,
        label="M = 0"
    )

    plt.xlabel(
        "Margin M = y(w^T x + b)"
    )

    plt.ylabel(
        "Количество объектов"
    )

    plt.title(
        "Распределение отступов"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "02_margins.png",
        dpi=160
    )

    plt.close()


def plot_objective(
    models
):
    plt.figure(
        figsize=(
            11,
            6
        )
    )

    for name in models:

        model = models[
            name
        ]

        plt.plot(
            model.history[
                "objective"
            ],
            label=name
        )

    plt.xlabel(
        "Epoch / iteration"
    )

    plt.ylabel(
        "Q"
    )

    plt.title(
        "Сходимость методов обучения"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "03_objective.png",
        dpi=160
    )

    plt.close()


def plot_recurrent_quality(
    model
):
    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        model.history[
            "objective"
        ],
        label="Полный Q"
    )

    plt.plot(
        model.history[
            "recurrent_objective"
        ],
        label="Рекуррентный Q"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Quality functional"
    )

    plt.title(
        "Рекуррентная оценка функционала"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "04_recurrent_quality.png",
        dpi=160
    )

    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred
):
    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(
            6,
            5
        )
    )

    plt.imshow(
        matrix
    )

    plt.xticks(
        [0, 1],
        ["-1", "+1"]
    )

    plt.yticks(
        [0, 1],
        ["-1", "+1"]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        "Confusion Matrix"
    )

    for row in range(
        2
    ):
        for column in range(
            2
        ):
            plt.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        "05_confusion_matrix.png",
        dpi=160
    )

    plt.close()