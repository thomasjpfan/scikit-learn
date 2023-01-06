import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import pytest

from sklearn.preprocessing import TargetRegressorEncoder
from sklearn.model_selection import KFold


@pytest.mark.parametrize(
    "categories",
    [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, 3.0, np.nan], dtype=np.float64),
        np.array(["cat", "dog", "snake"], dtype=object),
        "auto",
    ],
)
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("smooth", [5.0, 10.0])
def test_regression(categories, seed, smooth):
    """Check regression encoding."""

    X_int = np.array([[0] * 20 + [1] * 30 + [2] * 40], dtype=np.int64).T
    n_samples = X_int.shape[0]

    if isinstance(categories, str) and categories == "auto":
        X_input = X_int
    else:
        X_input = categories[X_int]

    rng = np.random.RandomState(seed)
    y = rng.uniform(low=-10, high=20, size=n_samples)

    # compute encodings for all data to validate `transform`
    y_mean = np.mean(y)
    smooth_sum = smooth * y_mean

    expected_encoding_0 = (np.sum(y[:20]) + smooth_sum) / (20.0 + smooth)
    expected_encoding_1 = (np.sum(y[20:50]) + smooth_sum) / (30.0 + smooth)
    expected_encoding_2 = (np.sum(y[50:]) + smooth_sum) / (40.0 + smooth)
    expected_encodings = np.asarray(
        [expected_encoding_0, expected_encoding_1, expected_encoding_2]
    )

    shuffled_idx = rng.permutation(n_samples)
    X_int = X_int[shuffled_idx]
    X_input = X_input[shuffled_idx]
    y = y[shuffled_idx]

    # Get encodings for cv splits to validate `fit_transform`
    expected_X_fit_transform = np.empty_like(X_int, dtype=np.float64)
    kfold = KFold(n_splits=3)
    for train_idx, test_idx in kfold.split(X_input):
        cur_encodings = np.zeros(3, dtype=np.float64)
        X_, y_ = X_int[train_idx, :], y[train_idx]
        for c in range(3):
            y_subset = y_[X_[:, 0] == c]
            current_sum = np.sum(y_subset) + smooth_sum
            current_cnt = y_subset.shape[0] + smooth
            cur_encodings[c] = current_sum / current_cnt

            expected_X_fit_transform[test_idx, 0] = np.take(
                cur_encodings, indices=X_input[test_idx, 0]
            )

    target_encoder = TargetRegressorEncoder(
        smooth=smooth, categories=categories, cv=kfold
    )

    X_fit_transform = target_encoder.fit_tranfsorm(X_input, y)
    assert_allclose(X_fit_transform, expected_X_fit_transform)
    assert len(target_encoder.encodings_) == 1
    assert_allclose(target_encoder.encodings_[0], expected_encodings)
    assert target_encoder.encoding_mean_ == pytest.approx(y_mean)

    # 3 is unknown and will be encoded as the mean
    X_test = np.array([[0, 1, 2, 3]], dtype=np.int64).T
    expected_X_test_transform = np.concatenate(
        (expected_encodings, np.asarray([y_mean]))
    ).reshape(-1, 1)

    if isinstance(categories, str) and categories == "auto":
        X_test_input = X_test
    else:
        X_test_input = categories[X_test]

    X_test_transform = target_encoder.transform(X_test_input)
    assert_allclose(X_test_transform, expected_X_test_transform)


@pytest.mark.parametrize(
    "X, categories",
    [
        (
            np.array([[0] * 10 + [1] * 10 + [3]], dtype=np.int64).T,  # 3 is unknown
            [[0, 1, 2]],
        ),
        (
            np.array(
                [["cat"] * 10 + ["dog"] * 10 + ["snake"]], dtype=object
            ).T,  # snake is unknown
            [["dog", "cat", "cow"]],
        ),
    ],
)
def test_regression_custom_categories(X, categories):
    """custom categoires with unknown categories that are not in training data."""
    rng = np.random.RandomState(42)
    y = rng.uniform(low=-10, high=20, size=X.shape[0])
    y_mean = y.mean()

    enc = TargetRegressorEncoder(categories=categories)
    X_fit_trans = enc.fit_transform(X, y)

    # The last element is unknown and encoded as the mean
    assert_allclose(X_fit_trans[-1], [y_mean])

    X_trans = enc.transform([X[0, -1]])
    assert X_trans[0, 0] == pytest.approx(y_mean)

    assert len(enc.encodings_) == 1
    # custom category that is not in training data
    assert enc.encodings_[0][-1] == pytest.approx(y_mean)


@pytest.mark.parametrize(
    "y, msg",
    [
        ([1, 2, 0, 1], "Found input variables with inconsistent"),
        (["cat", "dog", "bear"], "dtype='numeric' is not compatible"),
    ],
)
def test_regression_errors(y, msg):
    """Check invalidate input."""
    X = np.asarray([[1, 0, 1]]).T

    enc = TargetRegressorEncoder()
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X, y)


def test_regression_feature_names_out_set_output():
    """Check TargetRegressorEncoder works with set_output."""
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"] * 10, "B": [1, 2] * 10})
    y = [1, 2] * 10

    enc_default = TargetRegressorEncoder(cv=2, smooth=3.0).set_output(
        transform="default"
    )
    enc_pandas = TargetRegressorEncoder(cv=2, smooth=3.0).set_output(transform="pandas")

    X_default = enc_default.fit_transform(X_df, y)
    X_pandas = enc_pandas.fit_transform(X_df, y)

    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(enc_pandas.get_feature_names_out(), X_pandas.columns)


@pytest.mark.parametrize("to_pandas", [True, False])
def test_regression_multiple_features_quick(to_pandas):
    """Check regression encoder with multiple features."""
    X = np.array(
        [[1, 1], [0, 1], [1, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int64
    )
    # y = np.array([0, 1, 2, 3, 4, 5, 10, 7])
    # y_mean = np.mean(y)

    X_test = np.array(
        [
            [0, 1],
            [1, 0],
            [2, 10],  # unknown
        ],
        dtype=int,
    )

    if to_pandas:
        pd = pytest.importorskip("pandas")
        # convert second feature to a object
        X_obj = np.array(["cat", "dog"], dtype=object)[X[:, 1]]
        X = pd.DataFrame({"feat0": X[:, 0], "feat1": X_obj}, columns=["feat0", "feat1"])
        X_test = pd.DataFrame({"feat0": X_test[:, 0], "feat1": ["dog", "cat", "snake"]})

    # manually compute encoding
