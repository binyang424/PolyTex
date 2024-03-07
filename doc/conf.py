# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('./../polytex'))  # 指向src目录

# -- Project information -----------------------------------------------------
# github
project = 'PolyTex'
copyright = '2022-2024, Bin Yang'
author = 'Bin Yang'
release = '0.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              # "sphinx.ext.napoleon",
              'sphinx_simplepdf',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary', # Create neat summary tables for modules/classes/methods etc
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx_search.extension',
              "numpydoc",
              'sphinx.ext.graphviz',
              'sphinx.ext.inheritance_diagram',
              'sphinx_copybutton',
              'matplotlib.sphinxext.plot_directive',
              "m2r2",
              "sphinx_gallery.gen_gallery",
              ]

warning_is_error = False

# autosummaries from source-files
autodoc_default_flags = ["members", "inherited-members"]
autosummary_generate = True
autosummary_imported_members = True

# dont show __init__ docstring
autoclass_content = "init"
# sort class members
autodoc_member_order = "groupwise"
# autodoc_member_order = 'bysource'

inheritance_graph_attrs = dict(rankdir="TB", size='""')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../test",
    # path where to save gallery generated examples
    "gallery_dirs": "source/test",
    "filename_pattern": "/.*.py",
    'only_warn_on_example_error': True
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- MyST-Parser/MyST-NB configuration ------------------------------------------------------
myst_heading_anchors = 4
nb_execution_mode = 'off'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = 'sphinx_rtd_theme'
html_static_path = []
html_logo = "source/polytex_logo.png"
html_theme_options = {
    'logo_only': False,
    'display_version': True,
}
