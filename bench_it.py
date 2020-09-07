from sklearn.datasets import make_classification
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from memory_profiler import memory_usage

X, y = make_classification(n_classes=2,
                           n_samples=10_000,
                           n_features=400,
                           random_state=0)

hgb = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=127,
    learning_rate=.1,
    random_state=0,
    verbose=1,
)

mems = memory_usage((hgb.fit, (X, y)))
print(f"{max(mems):.2f}, {max(mems) - min(mems):.2f} MB")
