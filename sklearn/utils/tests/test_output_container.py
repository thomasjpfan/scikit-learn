import pytest

import numpy as np
from scipy.sparse import bsr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
from numpy.testing import assert_array_equal


from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.tests.test_text import JUNK_FOOD_DOCS
from sklearn.feature_extraction.tests.test_text import NOTJUNK_FOOD_DOCS

from sklearn.utils.output_container import NamedBSRMatrix
from sklearn.utils.output_container import NamedCOOMatrix
from sklearn.utils.output_container import NamedCSRMatrix
from sklearn.utils.output_container import NamedCSCMatrix
from sklearn.utils.output_container import NamedDIAMatrix
from sklearn.utils.output_container import NamedDOKMatrix
from sklearn.utils.output_container import NamedLILMatrix
from sklearn.utils.output_container import make_named_container


@pytest.mark.parametrize(
    "scipy_sparse, NamedSparseMatrix",
    [
        (bsr_matrix, NamedBSRMatrix),
        (coo_matrix, NamedCOOMatrix),
        (csc_matrix, NamedCSCMatrix),
        (csr_matrix, NamedCSRMatrix),
        (dia_matrix, NamedDIAMatrix),
        (dok_matrix, NamedDOKMatrix),
        (lil_matrix, NamedLILMatrix),
    ],
)
def test_make_named_container_sparse(scipy_sparse, NamedSparseMatrix):
    """Check make_named_container for sparse data."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    sparse_X = scipy_sparse(X)

    columns = np.asarray(["f0", "f1", "f2"], dtype=object)
    index = np.asarray([0, 1])

    # default arguments return a no-op
    sparse_default = make_named_container(sparse_X)
    assert sparse_default is sparse_X

    sparse_named = make_named_container(
        sparse_X,
        get_columns=lambda: columns,
        sparse_container="namedsparse",
        index=index,
    )

    assert isinstance(sparse_named, scipy_sparse)
    assert sparse_named.__class__ == NamedSparseMatrix
    assert sparse_named.columns is columns
    assert sparse_named.index is index

    # Updates columns if the input is named sparse
    new_columns = np.asarray(["g0", "g1", "g2"], dtype=object)
    spared_named_rt = make_named_container(
        sparse_named, sparse_container="namedsparse", get_columns=lambda: new_columns
    )
    assert spared_named_rt is sparse_named
    assert_array_equal(spared_named_rt.columns, new_columns)


def test_make_named_container_dense():
    """Check make_named_container for dense data."""
    pd = pytest.importorskip("pandas")
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    columns = np.asarray(["f0", "f1", "f2"], dtype=object)
    index = np.asarray([0, 1])

    X_default = make_named_container(X)
    assert X_default is X

    dense_named = make_named_container(
        X, get_columns=lambda: columns, dense_container="pandas", index=index
    )
    assert isinstance(dense_named, pd.DataFrame)
    assert_array_equal(dense_named.columns, columns)
    assert_array_equal(dense_named.index, index)

    # Updates dataframe columns if the input is a dataframe
    new_columns = np.asarray(["g0", "g1", "g2"], dtype=object)
    dense_named_rt = make_named_container(
        dense_named, dense_container="pandas", get_columns=lambda: new_columns
    )
    assert dense_named_rt is dense_named
    assert_array_equal(dense_named_rt.columns, new_columns)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(dense_container="invalid"), "dense_container must be 'default' or"),
        (dict(sparse_container="invalid"), "sparse_container must be 'default' or"),
    ],
)
def test_make_named_container_error_validation(kwargs, match):
    X = np.asarray([[1, 0, 3], [0, 0, 1]])

    with pytest.raises(ValueError, match=match):
        make_named_container(X, **kwargs)


def test_text_data_integration():
    """Integration test for text data."""
    ALL_FOOD_DOCS = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS
    y = [0] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)
    pipe = make_pipeline(CountVectorizer(), TfidfTransformer(), LogisticRegression())
    pipe.set_output(transform="pandas_or_namedsparse")

    pipe.fit(ALL_FOOD_DOCS, y)

    X_trans = pipe[:-1].transform(ALL_FOOD_DOCS)
    features_in = pipe[:-1].get_feature_names_out()
    assert_array_equal(features_in, pipe[-1].feature_names_in_)
    assert_array_equal(features_in, X_trans.columns)
