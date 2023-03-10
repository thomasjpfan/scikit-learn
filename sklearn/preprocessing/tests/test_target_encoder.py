import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import pytest

from sklearn.preprocessing import TargetEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def _encode_target(X_ordinal, y_int, n_categories, smooth):
    """Simple Python implementation of target encoding."""
    cur_encodings = np.zeros(n_categories, dtype=np.float64)
    y_mean = np.mean(y_int)

    if smooth == "auto":
        y_variance = np.var(y_int)
        for c in range(n_categories):
            y_subset = y_int[X_ordinal == c]
            n_i = y_subset.shape[0]

            if n_i == 0:
                cur_encodings[c] = y_mean
                continue

            y_subset_variance = np.var(y_subset)
            m = y_subset_variance / y_variance
            lambda_ = n_i / (n_i + m)

            cur_encodings[c] = lambda_ * np.mean(y_subset) + (1 - lambda_) * y_mean
        return cur_encodings
    else:  # float
        for c in range(n_categories):
            y_subset = y_int[X_ordinal == c]
            current_sum = np.sum(y_subset) + y_mean * smooth
            current_cnt = y_subset.shape[0] + smooth
            cur_encodings[c] = current_sum / current_cnt
        return cur_encodings


@pytest.mark.parametrize(
    "categories, unknown_value",
    [
        ([np.array([0, 1, 2], dtype=np.int64)], 4),
        ([np.array([1.0, 3.0, np.nan], dtype=np.float64)], 6.0),
        ([np.array(["cat", "dog", "snake"], dtype=object)], "bear"),
        ("auto", 3),
    ],
)
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("smooth", [5.0, "auto"])
@pytest.mark.parametrize("target_type", ["binary", "continuous"])
def test_encoding(categories, unknown_value, seed, smooth, target_type):
    """Check encoding for binary and continuous targets."""

    X_ordinal = np.array([[0] * 20 + [1] * 30 + [2] * 40], dtype=np.int64).T
    n_categories = 3
    n_samples = X_ordinal.shape[0]

    if isinstance(categories, str) and categories == "auto":
        X_input = X_ordinal
    else:
        X_input = categories[0][X_ordinal]

    rng = np.random.RandomState(seed)

    if target_type == "binary":
        y_int = rng.randint(low=0, high=2, size=n_samples)
        target_names = np.array(["cat", "dog"], dtype=object)
        y_input = target_names[y_int]
        cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    else:  # target_type == continuous
        y_int = rng.uniform(low=-10, high=20, size=n_samples)
        y_input = y_int
        cv = KFold(n_splits=3, random_state=0, shuffle=True)

    shuffled_idx = rng.permutation(n_samples)
    X_ordinal = X_ordinal[shuffled_idx]
    X_input = X_input[shuffled_idx]
    y_input = y_input[shuffled_idx]
    y_int = y_int[shuffled_idx]

    # Get encodings for cv splits to validate `fit_transform`
    expected_X_fit_transform = np.empty_like(X_ordinal, dtype=np.float64)

    for train_idx, test_idx in cv.split(X_ordinal, y_input):
        X_, y_ = X_ordinal[train_idx, 0], y_int[train_idx]
        cur_encodings = _encode_target(X_, y_, n_categories, smooth)
        expected_X_fit_transform[test_idx, 0] = cur_encodings[X_ordinal[test_idx, 0]]

    target_encoder = TargetEncoder(
        smooth=smooth, categories=categories, cv=3, random_state=0
    )

    X_fit_transform = target_encoder.fit_transform(X_input, y_input)

    assert target_encoder.target_type_ == target_type
    assert_allclose(X_fit_transform, expected_X_fit_transform)
    assert len(target_encoder.encodings_) == 1

    # compute encodings for all data to validate `transform`
    y_mean = np.mean(y_int)
    expected_encodings = _encode_target(X_ordinal[:, 0], y_int, n_categories, smooth)
    assert_allclose(target_encoder.encodings_[0], expected_encodings)
    assert target_encoder.target_mean_ == pytest.approx(y_mean)

    X_test = np.array([[0, 1, 2]], dtype=np.int64).T
    expected_X_test_transform = np.concatenate(
        (expected_encodings, np.array([y_mean]))
    ).reshape(-1, 1)

    if isinstance(categories, str) and categories == "auto":
        X_test_input = X_test
    else:
        X_test_input = categories[0][X_test]

    X_test_input = np.concatenate((X_test_input, [[unknown_value]]))

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
@pytest.mark.parametrize("smooth", [4.0, "auto"])
def test_custom_categories(X, categories, smooth):
    """Custom categories with unknown categories that are not in training data."""
    rng = np.random.RandomState(0)
    y = rng.uniform(low=-10, high=20, size=X.shape[0])
    enc = TargetEncoder(categories=categories, smooth=smooth, random_state=0).fit(X, y)

    # The last element is unknown and encoded as the mean
    y_mean = y.mean()
    X_trans = enc.transform(X[-1:])
    assert X_trans[0, 0] == pytest.approx(y_mean)

    assert len(enc.encodings_) == 1
    # custom category that is not in training data
    assert enc.encodings_[0][-1] == pytest.approx(y_mean)


@pytest.mark.parametrize(
    "y, msg",
    [
        ([1, 2, 0, 1], "Found input variables with inconsistent"),
        (
            np.array([[1, 2, 0], [1, 2, 3]]).T,
            "Target type was inferred to be 'multiclass-multioutput'",
        ),
        (["cat", "dog", "bear"], "Target type was inferred to be 'multiclass'"),
    ],
)
def test_errors(y, msg):
    """Check invalidate input."""
    X = np.array([[1, 0, 1]]).T

    enc = TargetEncoder()
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X, y)


def test_use_regression_target():
    """Custom target_type to avoid inferring the target type."""
    X = np.array([[0, 1, 0, 1, 0, 1]]).T

    # XXX: When multiclass is supported, then the following `y`
    # is considered a multiclass problem and `TargetEncoder` will not error.
    # type_of_target would be 'multiclass'
    y = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    enc = TargetEncoder()
    msg = "Target type was inferred to be 'multiclass'"
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X, y)

    enc = TargetEncoder(target_type="continuous")
    enc.fit_transform(X, y)
    assert enc.target_type_ == "continuous"


def test_feature_names_out_set_output():
    """Check TargetEncoder works with set_output."""
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"] * 10, "B": [1, 2] * 10})
    y = [1, 2] * 10

    enc_default = TargetEncoder(cv=2, smooth=3.0, random_state=0)
    enc_default.set_output(transform="default")
    enc_pandas = TargetEncoder(cv=2, smooth=3.0, random_state=0)
    enc_pandas.set_output(transform="pandas")

    X_default = enc_default.fit_transform(X_df, y)
    X_pandas = enc_pandas.fit_transform(X_df, y)

    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(enc_pandas.get_feature_names_out(), ["A", "B"])
    assert_array_equal(enc_pandas.get_feature_names_out(), X_pandas.columns)


@pytest.mark.parametrize("to_pandas", [True, False])
@pytest.mark.parametrize("smooth", [1.0, "auto"])
@pytest.mark.parametrize("target_type", ["binary-ints", "binary-str", "continuous"])
def test_multiple_features_quick(to_pandas, smooth, target_type):
    """Check target encoder with multiple features."""
    X_ordinal = np.array(
        [[1, 1], [0, 1], [1, 1], [2, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int64
    )
    if target_type == "binary-str":
        y_input = np.array(["a", "b", "a", "a", "b", "b", "a", "b"])
        y_int = LabelEncoder().fit_transform(y_input)
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    elif target_type == "binary-ints":
        y_input = np.array([3, 4, 3, 3, 3, 4, 4, 4])
        y_int = LabelEncoder().fit_transform(y_input)
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    else:
        y_input = np.array([3.0, 5.1, 2.4, 3.5, 4.1, 5.5, 10.3, 7.3], dtype=np.float32)
        y_int = y_input
        cv = KFold(2, random_state=0, shuffle=True)
    y_mean = np.mean(y_int)
    categories = [[0, 1, 2], [0, 1]]

    X_test = np.array(
        [
            [0, 1],
            [3, 0],  # 3 is unknown
            [1, 10],  # 10 is unknown
        ],
        dtype=np.int64,
    )

    if to_pandas:
        pd = pytest.importorskip("pandas")
        # convert second feature to an object
        X_input = pd.DataFrame(
            {
                "feat0": X_ordinal[:, 0],
                "feat1": np.array(["cat", "dog"], dtype=object)[X_ordinal[:, 1]],
            }
        )
        # "snake" is unknown
        X_test = pd.DataFrame({"feat0": X_test[:, 0], "feat1": ["dog", "cat", "snake"]})
    else:
        X_input = X_ordinal

    # manually compute encoding for fit_transform
    expected_X_fit_transform = np.empty_like(X_ordinal, dtype=np.float64)
    for f_idx, cats in enumerate(categories):
        for train_idx, test_idx in cv.split(X_ordinal, y_int):
            X_, y_ = X_ordinal[train_idx, f_idx], y_int[train_idx]
            current_encoding = _encode_target(X_, y_, len(cats), smooth)
            expected_X_fit_transform[test_idx, f_idx] = current_encoding[
                X_ordinal[test_idx, f_idx]
            ]

    # manually compute encoding for transform
    expected_encodings = []
    for f_idx, cats in enumerate(categories):
        current_encoding = _encode_target(X_ordinal[:, f_idx], y_int, len(cats), smooth)
        expected_encodings.append(current_encoding)

    expected_X_test_transform = np.array(
        [
            [expected_encodings[0][0], expected_encodings[1][1]],
            [y_mean, expected_encodings[1][0]],
            [expected_encodings[0][1], y_mean],
        ],
        dtype=np.float64,
    )

    enc = TargetEncoder(smooth=smooth, cv=2, random_state=0)
    X_fit_transform = enc.fit_transform(X_input, y_input)
    assert_allclose(X_fit_transform, expected_X_fit_transform)

    assert len(enc.encodings_) == 2
    for i in range(2):
        assert_allclose(enc.encodings_[i], expected_encodings[i])

    X_test_transform = enc.transform(X_test)
    assert_allclose(X_test_transform, expected_X_test_transform)


@pytest.mark.parametrize(
    "y, y_mean",
    [
        (np.array([3.4] * 20), 3.4),
        (np.array([0] * 20), 0),
        (np.array(["a"] * 20, dtype=object), 0),
    ],
    ids=["continuous", "binary", "binary-string"],
)
@pytest.mark.parametrize("smooth", ["auto", 4])
def test_constant_target_and_feature(y, y_mean, smooth):
    """Check edge case where feature and target is constant."""
    X = np.array([[1] * 20]).T
    n_samples = X.shape[0]

    enc = TargetEncoder(cv=2, smooth=smooth, random_state=0)
    X_trans = enc.fit_transform(X, y)
    assert_allclose(X_trans, np.repeat([[y_mean]], n_samples, axis=0))
    assert enc.encodings_[0][0] == pytest.approx(y_mean)
    assert enc.target_mean_ == pytest.approx(y_mean)

    X_test = np.array([[1], [0]])
    X_test_trans = enc.transform(X_test)
    assert_allclose(X_test_trans, np.repeat([[y_mean]], 2, axis=0))


def test_fit_transform_not_associated_withy_if_ordinal_categorical_is_not(
    global_random_seed,
):
    cardinality = 30  # not too large, otherwise we need a very large n_samples
    n_samples = 3000
    rng = np.random.RandomState(global_random_seed)
    y_train = rng.normal(size=n_samples)
    X_train = rng.randint(0, cardinality, size=n_samples).reshape(-1, 1)

    # Sort by y_train to attempt to cause a leak
    y_sorted_indices = y_train.argsort()
    y_train = y_train[y_sorted_indices]
    X_train = X_train[y_sorted_indices]

    target_encoder = TargetEncoder(shuffle=True, random_state=global_random_seed)
    X_encoded_train_shuffled = target_encoder.fit_transform(X_train, y_train)

    target_encoder = TargetEncoder(shuffle=False)
    X_encoded_train_no_shuffled = target_encoder.fit_transform(X_train, y_train)

    # Check that no information about y_train has leaked into X_train:
    regressor = RandomForestRegressor(
        n_estimators=10, min_samples_leaf=20, random_state=global_random_seed
    )

    # It's impossible to learn a good predictive model on the training set when
    # using the original representation X_train or the target encoded
    # representation with shuffled inner CV. For the latter, no information
    # about y_train has inadvertently leaked into the prior used to generate
    # `X_encoded_train_shuffled`:
    cv = ShuffleSplit(n_splits=50, random_state=global_random_seed)
    assert cross_val_score(regressor, X_train, y_train, cv=cv).mean() < 0.1
    assert (
        cross_val_score(regressor, X_encoded_train_shuffled, y_train, cv=cv).mean()
        < 0.1
    )

    # Without the inner CV shuffling, a lot of information about y_train goes into the
    # the per-fold y_train.mean() priors: shrinkage is no longer effective in this
    # case and would no longer be able to prevent downstream over-fitting.
    assert (
        cross_val_score(regressor, X_encoded_train_no_shuffled, y_train, cv=cv).mean()
        > 0.5
    )


def test_smooth_zero():
    """Check edge case with zero smoothing and cv does not contain category."""
    X = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    y = np.array([2.1, 4.3, 1.2, 3.1, 1.0, 9.0, 10.3, 14.2, 13.3, 15.0])

    enc = TargetEncoder(smooth=0.0, shuffle=False, cv=2)
    X_trans = enc.fit_transform(X, y)

    # With cv = 2, category 0 does not exist in the second half, thus
    # it will be encoded as the mean of the second half
    assert_allclose(X_trans[0], np.mean(y[5:]))

    # category 1 does nto exist in the first half, thus it will be encoded as
    # the mean of the first half
    assert_allclose(X_trans[-1], np.mean(y[:5]))
