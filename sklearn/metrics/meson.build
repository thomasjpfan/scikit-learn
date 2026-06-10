# Metrics is cimported from other subpackages so this is needed for the cimport
# to work
metrics_cython_tree = [
  fs.copyfile('__init__.py')
]
# Some metrics code cimports code from utils, we may as well copy all the necessary files
metrics_cython_tree += utils_cython_tree

# _dist_metrics.pxd is cimported by other subpackages (e.g. sklearn.cluster,
# sklearn.neighbors), so copy it into the build tree and ship it.
metrics_cython_tree += [fs.copyfile('_dist_metrics.pxd')]
py.install_sources('_dist_metrics.pxd', subdir: 'sklearn/metrics')

# _dist_metrics is compiled as C++ because its DistanceMetric class delegates
# its vector-wise math to the C++ functors in src/metric_kernels.hpp.
_dist_metrics = py.extension_module(
  '_dist_metrics',
  [cython_gen_cpp.process('_dist_metrics.pyx'), metrics_cython_tree],
  dependencies: [np_dep],
  include_directories: include_directories('src'),
  subdir: 'sklearn/metrics',
  install: true
)

py.extension_module(
  '_pairwise_fast',
  [cython_gen.process('_pairwise_fast.pyx'), metrics_cython_tree],
  dependencies: [openmp_dep],
  subdir: 'sklearn/metrics',
  install: true
)

subdir('_pairwise_distances_reduction')
subdir('cluster')
