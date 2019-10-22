"""Generates submodule to allow deprecation of submodules and keeping git
blame."""
from pathlib import Path
from contextlib import suppress

# TODO: Remove the whole file in 0.24

# This is a set of 4-tuples consisting of
# (new_module_name, deprecated_path, correct_import_path, importee)
# importee is used by test_import_deprecations to check for DeprecationWarnings
_DEPRECATED_MODULES = [
    ('_mocking', 'sklearn.utils.mocking', 'sklearn.utils',
     'MockDataFrame'),

    ('_bagging', 'sklearn.ensemble.bagging', 'sklearn.ensemble',
     'BaggingClassifier'),
    ('_base', 'sklearn.ensemble.base', 'sklearn.ensemble',
     'BaseEnsemble'),
    ('_forest', 'sklearn.ensemble.forest', 'sklearn.ensemble',
     'RandomForestClassifier'),
    ('_gb', 'sklearn.ensemble.gradient_boosting', 'sklearn.ensemble',
     'GradientBoostingClassifier'),
    ('_iforest', 'sklearn.ensemble.iforest', 'sklearn.ensemble',
     'IsolationForest'),
    ('_voting', 'sklearn.ensemble.voting', 'sklearn.ensemble',
     'VotingClassifier'),
    ('_weight_boosting', 'sklearn.ensemble.weight_boosting',
     'sklearn.ensemble', 'AdaBoostClassifier'),
    ('_classes', 'sklearn.tree.tree', 'sklearn.tree',
     'DecisionTreeClassifier'),
    ('_export', 'sklearn.tree.export', 'sklearn.tree', 'export_graphviz'),

    ('_rbm', 'sklearn.neural_network.rbm', 'sklearn.neural_network',
     'BernoulliRBM'),
    ('_multilayer_perceptron', 'sklearn.neural_network.multilayer_perceptron',
     'sklearn.neural_network', 'MLPClassifier'),

    ('_weight_vector', 'sklearn.utils.weight_vector', 'sklearn.utils',
     'WeightVector'),
    ('_seq_dataset', 'sklearn.utils.seq_dataset', 'sklearn.utils',
     'ArrayDataset32'),
    ('_fast_dict', 'sklearn.utils.fast_dict', 'sklearn.utils', 'IntFloatDict'),

    ('_affinity_propagation', 'sklearn.cluster.affinity_propagation_',
     'sklearn.cluster', 'AffinityPropagation'),
    ('_bicluster', 'sklearn.cluster.bicluster', 'sklearn.cluster',
     'SpectralBiclustering'),
    ('_birch', 'sklearn.cluster.birch', 'sklearn.cluster', 'Birch'),
    ('_dbscan', 'sklearn.cluster.dbscan_', 'sklearn.cluster', 'DBSCAN'),
    ('_hierarchical', 'sklearn.cluster.hierarchical', 'sklearn.cluster',
     'FeatureAgglomeration'),
    ('_k_means', 'sklearn.cluster.k_means_', 'sklearn.cluster', 'KMeans'),
    ('_mean_shift', 'sklearn.cluster.mean_shift_', 'sklearn.cluster',
     'MeanShift'),
    ('_optics', 'sklearn.cluster.optics_', 'sklearn.cluster', 'OPTICS'),
    ('_spectral', 'sklearn.cluster.spectral', 'sklearn.cluster',
     'SpectralClustering'),

    ('_base', 'sklearn.mixture.base', 'sklearn.mixture', 'BaseMixture'),
    ('_gaussian_mixture', 'sklearn.mixture.gaussian_mixture',
     'sklearn.mixture', 'GaussianMixture'),
    ('_bayesian_mixture', 'sklearn.mixture.bayesian_mixture',
     'sklearn.mixture', 'BayesianGaussianMixture'),

    ('_empirical_covariance_', 'sklearn.covariance.empirical_covariance_',
     'sklearn.covariance', 'EmpiricalCovariance'),
    ('_shrunk_covariance_', 'sklearn.covariance.shrunk_covariance_',
     'sklearn.covariance', 'ShrunkCovariance'),
    ('_robust_covariance', 'sklearn.covariance.robust_covariance',
     'sklearn.covariance', 'MinCovDet'),
    ('_graph_lasso_', 'sklearn.covariance.graph_lasso_',
     'sklearn.covariance', 'GraphicalLasso'),
    ('_elliptic_envelope', 'sklearn.covariance.elliptic_envelope',
     'sklearn.covariance', 'EllipticEnvelope'),

    ('_cca_', 'sklearn.cross_decomposition.cca_',
     'sklearn.cross_decomposition', 'CCA'),
    ('_pls_', 'sklearn.cross_decomposition.pls_',
     'sklearn.cross_decomposition', 'PLSSVD'),

    ('_base', 'sklearn.svm.base', 'sklearn.svm', 'BaseLibSVM'),
    ('_bounds', 'sklearn.svm.bounds', 'sklearn.svm', 'l1_min_c'),
    ('_classes', 'sklearn.svm.classes', 'sklearn.svm', 'SVR'),
    ('_libsvm', 'sklearn.svm.libsvm', 'sklearn.svm', 'fit'),
    ('_libsvm_sparse', 'sklearn.svm.libsvm_sparse', 'sklearn.svm',
     'set_verbosity_wrap'),
    ('_liblinear', 'sklearn.svm.liblinear', 'sklearn.svm', 'train_wrap'),

    ('_base', 'sklearn.linear_model.base', 'sklearn.linear_model',
     'LinearRegression'),
    ('_cd_fast', 'sklearn.linear_model.cd_fast', 'sklearn.linear_model',
     'sparse_enet_coordinate_descent'),
    ('_bayes', 'sklearn.linear_model.bayes', 'sklearn.linear_model',
     'BayesianRidge'),
    ('_coordinate_descent', 'sklearn.linear_model.coordinate_descent',
     'sklearn.linear_model', 'Lasso'),
    ('_huber', 'sklearn.linear_model.huber', 'sklearn.linear_model',
     'HuberRegressor'),
    ('_least_angle', 'sklearn.linear_model.least_angle',
     'sklearn.linear_model', 'LassoLarsCV'),
    ('_logistic', 'sklearn.linear_model.logistic', 'sklearn.linear_model',
     'LogisticRegression'),
    ('_omp', 'sklearn.linear_model.omp', 'sklearn.linear_model',
     'OrthogonalMatchingPursuit'),
    ('_passive_aggressive', 'sklearn.linear_model.passive_aggressive',
     'sklearn.linear_model', 'PassiveAggressiveClassifier'),
    ('_perceptron', 'sklearn.linear_model.perceptron', 'sklearn.linear_model',
     'Perceptron'),
    ('_ransac', 'sklearn.linear_model.ransac', 'sklearn.linear_model',
     'RANSACRegressor'),
    ('_ridge', 'sklearn.linear_model.ridge', 'sklearn.linear_model',
     'Ridge'),
    ('_sag', 'sklearn.linear_model.sag', 'sklearn.linear_model',
     'get_auto_step_size'),
    ('_sag_fast', 'sklearn.linear_model.sag_fast', 'sklearn.linear_model',
     'MultinomialLogLoss64'),
    ('_sgd_fast', 'sklearn.linear_model.sgd_fast', 'sklearn.linear_model',
     'Hinge'),
    ('_stochastic_gradient', 'sklearn.linear_model.stochastic_gradient',
     'sklearn.linear_model', 'SGDClassifier'),
    ('_theil_sen', 'sklearn.linear_model.theil_sen', 'sklearn.linear_model',
     'TheilSenRegressor'),
]


_FILE_CONTENT_TEMPLATE = """
# THIS FILE WAS AUTOMATICALLY GENERATED BY deprecated_modules.py

from .{new_module_name} import *  # noqa
from {relative_dots}utils.deprecation import _raise_dep_warning_if_not_pytest

deprecated_path = '{deprecated_path}'
correct_import_path = '{correct_import_path}'

_raise_dep_warning_if_not_pytest(deprecated_path, correct_import_path)
"""


def _get_deprecated_path(deprecated_path):
    deprecated_parts = deprecated_path.split(".")
    deprecated_parts[-1] = deprecated_parts[-1] + ".py"
    return Path(*deprecated_parts)


def _create_deprecated_modules_files():
    """Add submodules that will be deprecated. A file is created based
    on the deprecated submodule's name. When this submodule is imported a
    deprecation warning will be raised.
    """
    for (new_module_name, deprecated_path,
         correct_import_path, _) in _DEPRECATED_MODULES:
        relative_dots = deprecated_path.count(".") * "."
        deprecated_content = _FILE_CONTENT_TEMPLATE.format(
            new_module_name=new_module_name,
            relative_dots=relative_dots,
            deprecated_path=deprecated_path,
            correct_import_path=correct_import_path)

        with _get_deprecated_path(deprecated_path).open('w') as f:
            f.write(deprecated_content)


def _clean_deprecated_modules_files():
    """Removes submodules created by _create_deprecated_modules_files."""
    for _, deprecated_path, _, _ in _DEPRECATED_MODULES:
        with suppress(FileNotFoundError):
            _get_deprecated_path(deprecated_path).unlink()


if __name__ == "__main__":
    _clean_deprecated_modules_files()
