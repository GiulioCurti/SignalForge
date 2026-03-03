# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

init_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SignalForge', '__init__.py'))
with open(init_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line.startswith("__version__ ="):
            # Extract the version string between single quotes
            start = line.find("'")
            end = line.rfind("'")
            if start != -1 and end != -1 and end > start:
                __version__ =  line[start+1:end]
                break

import sphinx_rtd_theme # read the docs theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SignalForge'
copyright = '2025, Giulio Curti'
author = 'Giulio Curti'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    
]

todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
