# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "gallimimus"
copyright = "2023, Sarus"
author = "Sarus"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
]


autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
    "class-doc-from": "class",
}

autodoc_typehints = "description"

autodoc_typehints_description_target = "documented_params"
autodoc_inherit_docstrings = False

autodoc_type_aliases = {
    "VariableDict": "flax.core.scope.VariableDict",
    "Embedding": "Embedding",
    "Module": "flax.linen.Module",
    "PRNGKeyArray": "jax.random.PRNGKeyArray",
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]