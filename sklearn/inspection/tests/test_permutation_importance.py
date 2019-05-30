import pytest
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize("load_dataset,RandomForest", [
    (load_boston, RandomForestRegressor),
    (load_iris, RandomForestClassifier)
])
@pytest.mark.parametrize("n_rounds", [3, 5])
def test_permutation_importance_correlated_feature_regression(
        load_dataset, RandomForest, n_rounds):
    # Make sure that feature highly correlated to the target have a higher
    # importance
    rng = np.random.RandomState(42)

    dataset = load_dataset()
    X, y = dataset.data, dataset.target
    y_with_little_noise = (
        y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)

    X = np.hstack([X, y_with_little_noise])

    clf = RandomForest(n_estimators=10, random_state=42)
    clf.fit(X, y)

    permute_imp = permutation_importance(clf, X, y, n_rounds=n_rounds,
                                         random_state=rng)

    assert permute_imp.shape == (X.shape[1], n_rounds)

    permute_score_means = np.mean(permute_imp, axis=-1)

    # the correlated feature with y was added as the last column and should
    # have the highest importance
    assert np.all(permute_score_means[-1] > permute_score_means[:-1])


@pytest.mark.parametrize("load_dataset,RandomForest", [
    (load_boston, RandomForestRegressor),
    (load_iris, RandomForestClassifier)
])
@pytest.mark.parametrize("n_rounds", [3, 5])
def test_permutation_importance_correlated_feature_regression_pandas(
        load_dataset, RandomForest, n_rounds):
    pd = pytest.importorskip("pandas")

    # Make sure that feature highly correlated to the target have a higher
    # importance
    rng = np.random.RandomState(42)

    dataset = load_dataset()
    X, y = dataset.data, dataset.target
    y_with_little_noise = (
        y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)

    # Adds feature correlated with y as the last column
    X = pd.DataFrame(X, columns=dataset.feature_names)
    X['correlated_feature'] = y_with_little_noise

    clf = RandomForest(n_estimators=10, random_state=42)
    clf.fit(X, y)

    permute_imp = permutation_importance(clf, X, y, n_rounds=n_rounds,
                                         random_state=rng)

    assert permute_imp.shape == (X.shape[1], n_rounds)

    permute_score_means = np.mean(permute_imp, axis=-1)

    # the correlated feature with y was added as the last column and should
    # have the highest importance
    assert np.all(permute_score_means[-1] > permute_score_means[:-1])


@pytest.mark.parametrize("n_rounds", [3, 5])
def test_permutation_importance_mixed_types(n_rounds):
    rng = np.random.RandomState(42)

    # Last column is correlated with y
    X = np.array([[1.0, 2.0, 3.0, np.nan], ['a', 'b', 'a', 'b']]).T
    y = np.array([0, 1, 0, 1])

    clf = make_pipeline(OneHotEncoder(),
                        LogisticRegression(solver='lbfgs'))
    clf.fit(X, y)
    permute_imp = permutation_importance(clf, X, y, n_rounds=n_rounds,
                                         random_state=rng)

    assert permute_imp.shape == (X.shape[1], n_rounds)
    permute_score_means = np.mean(permute_imp, axis=-1)

    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(permute_score_means[-1] > permute_score_means[:-1])


@pytest.mark.parametrize("n_rounds", [3, 5])
def test_permutation_importance_mixed_types_pandas(n_rounds):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(42)

    # Last column is correlated with y
    X = np.array([[1.0, 2.0, 3.0, np.nan], ['a', 'b', 'a', 'b']]).T
    y = np.array([0, 1, 0, 1])

    X = pd.DataFrame(X, columns=['num_col', 'cat_col'])
    num_preprocess = make_pipeline(SimpleImputer(), StandardScaler())
    preprocess = ColumnTransformer([
        ('num', num_preprocess, ['num_col']),
        ('cat', OneHotEncoder(), ['cat_col'])
    ])
    clf = make_pipeline(preprocess, LogisticRegression(solver='lbfgs'))
    clf.fit(X, y)

    permute_imp = permutation_importance(clf, X, y, n_rounds=n_rounds,
                                         random_state=rng)

    assert permute_imp.shape == (X.shape[1], n_rounds)

    permute_score_means = np.mean(permute_imp, axis=-1)
    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(permute_score_means[-1] > permute_score_means[:-1])
