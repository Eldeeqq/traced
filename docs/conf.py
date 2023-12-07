# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('/Users/janperina/projects/masters-thesis'))

...

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Traced'
copyright = f'{date.today().year}, Jan Perina'
author = 'Jan Perina'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.githubpages',
    

]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
# html_static_path = ['_static']
