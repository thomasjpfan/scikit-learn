import pytest


def test_import_with_incorrect_name():
    msg = "cannot import name 'bad_name' from 'sklearn.ensemble'"
    with pytest.raises(ImportError, match=msg):
        from sklearn.ensemble import bad_name  # noqa
