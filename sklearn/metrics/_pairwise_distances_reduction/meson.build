# The pairwise-distances reductions are implemented in C++ and exposed with
# nanobind (see src/). The per-metric distance math is shared from
# sklearn/metrics/src/metric_kernels.hpp (../src).
_reductions = py.extension_module(
  '_reductions',
  'src/reductions.cpp',
  dependencies: [nanobind_dep, openmp_dep],
  include_directories: include_directories('src', '../src'),
  override_options: ['cpp_std=c++17'],
  gnu_symbol_visibility: 'hidden',
  subdir: 'sklearn/metrics/_pairwise_distances_reduction',
  install: true
)
