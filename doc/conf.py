# -*- coding: utf-8 -*-
#
# scikit-learn documentation build configuration file, created by
# sphinx-quickstart on Fri Jan  8 09:13:42 2010.
#
# This file is execfile()d with the current directory set to its containing
# dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import warnings
import re

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath('sphinxext'))

from github_link import make_linkcode_resolve
import sphinx_gallery

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.linkcode', 'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgconverter',
    'sphinx_gallery.gen_gallery',
    'sphinx_issues',
    'custom_references_resolver'
]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False


# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get('NO_MATHJAX'):
    extensions.append('sphinx.ext.imgmath')
    imgmath_image_format = 'svg'
else:
    extensions.append('sphinx.ext.mathjax')
    mathjax_path = ('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/'
                    'MathJax.js?config=TeX-AMS_SVG')


autodoc_default_options = {
    'members': True,
    'inherited-members': True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'contents'

# General information about the project.
project = 'scikit-learn'
copyright = '2007 - 2019, scikit-learn developers (BSD License)'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import sklearn
version = sklearn.__version__
# The full version, including alpha/beta/rc tags.
release = sklearn.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'templates', 'includes', 'themes']

# The reST default role (used for this markup: `text`) to use for all
# documents.
# sklearn uses a custom extension: `custom_references_resolver` to modify
# the order of link resolution for the 'any' role. It resolves python class
# links first before resolving 'std' domain links. Unresolved roles are
# considered to be <code> blocks.
default_role = 'any'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'scikit-learn-modern'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {'google_analytics': True,
                      'mathjax_path': mathjax_path}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['themes']


# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'scikit-learn'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'logos/scikit-learn-logo-notext.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'logos/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {'index': 'index.html',
                         'documentation': 'documentation.html'}

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'scikit-learndoc'


# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r"""
        \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bm}
        \usepackage{morefloats}\usepackage{enumitem} \setlistdepth{10}
        """
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto/manual]).
latex_documents = [('index', 'user_guide.tex', 'scikit-learn user guide',
                    'scikit-learn developers', 'manual'), ]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "logos/scikit-learn-logo.png"

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
latex_domain_indices = False

trim_doctests_flags = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
}

if 'dev' in version:
    binder_branch = 'master'
else:
    match = re.match(r'^(\d+)\.(\d+)(?:\.\d+)?$', version)
    if match is None:
        raise ValueError(
            'Ill-formed version: {!r}. Expected either '
            "a version containing 'dev' "
            'or a version like X.Y or X.Y.Z.'.format(version))

    major, minor = match.groups()
    binder_branch = '{}.{}.X'.format(major, minor)

sphinx_gallery_conf = {
    'doc_module': 'sklearn',
    'backreferences_dir': os.path.join('modules', 'generated'),
    'show_memory': True,
    'reference_url': {
        'sklearn': None},
    'examples_dirs': ['../examples'],
    'gallery_dirs': ['auto_examples'],
    'binder': {
        'org': 'scikit-learn',
        'repo': 'scikit-learn',
        'binderhub_url': 'https://mybinder.org',
        'branch': binder_branch,
        'dependencies': './binder/requirements.txt',
        'use_jupyter_lab': True
    }
}


# The following dictionary contains the information used to create the
# thumbnails for the front page of the scikit-learn home page.
# key: first image in set
# values: (number of plot in set, height of thumbnail)
carousel_thumbs = {'sphx_glr_plot_classifier_comparison_001.png': 600}


# enable experimental module so that experimental estimators can be
# discovered properly by sphinx
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.experimental import enable_iterative_imputer  # noqa


def make_carousel_thumbs(app, exception):
    """produces the final resized carousel images"""
    if exception is not None:
        return
    print('Preparing carousel images')

    image_dir = os.path.join(app.builder.outdir, '_images')
    for glr_plot, max_width in carousel_thumbs.items():
        image = os.path.join(image_dir, glr_plot)
        if os.path.exists(image):
            c_thumb = os.path.join(image_dir, glr_plot[:-4] + '_carousel.png')
            sphinx_gallery.gen_rst.scale_image(image, c_thumb, max_width, 190)


# Config for sphinx_issues

# we use the issues path for PRs since the issues URL will forward
issues_github_path = 'scikit-learn/scikit-learn'


def setup(app):
    # to hide/show the prompt in code examples:
    # app.add_javascript('js/copybutton.js')
    # app.add_javascript('js/extra.js')
    app.connect('build-finished', make_carousel_thumbs)


# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve('sklearn',
                                         'https://github.com/scikit-learn/'
                                         'scikit-learn/blob/{revision}/'
                                         '{package}/{path}#L{lineno}')

warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

# Reduces the output of estimators
sklearn.set_config(print_changed_only=True)
