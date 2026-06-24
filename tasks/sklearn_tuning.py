"""A realistic task: tune a gradient-boosting classifier on a real dataset.

Requires scikit-learn (``pip install scikit-learn``). Demonstrates that the
same loop that optimizes the toy quadratic also drives genuine model tuning —
the only thing that changes is the ``Task``.

    from autolab import Lab, AnthropicAgent
    from tasks.sklearn_tuning import GradientBoostingTuning

    lab = Lab(
        GradientBoostingTuning(),
        objective="accuracy",
        budget=25,
        agent=AnthropicAgent(),
    )
    print(lab.run().config)
"""

from __future__ import annotations

from autolab import Task


class GradientBoostingTuning(Task):
    """Tune a ``GradientBoostingClassifier`` on the breast-cancer dataset."""

    def propose_space(self):
        return {
            "n_estimators": (50, 400),
            "learning_rate": (1e-3, 5e-1),
            "max_depth": (1, 6),
            "subsample": (0.5, 1.0),
        }

    def run(self, config, seed):
        # Imported lazily so the package and its tests do not require sklearn.
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        X, y = load_breast_cancer(return_X_y=True)
        model = GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            subsample=config["subsample"],
            random_state=seed,
        )
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        return {"accuracy": float(scores.mean()), "accuracy_std": float(scores.std())}
