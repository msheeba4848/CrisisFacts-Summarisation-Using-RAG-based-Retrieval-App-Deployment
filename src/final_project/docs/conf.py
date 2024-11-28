# Configuration file for the Sphinx documentation builder.
#
# For a full list, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"final_project"
copyright = u"2024, Isfar Baset, Ziyan Di, Sheeba Moghal, Bella Shi and Jacky Zhang"
author = u"Isfar Baset, Ziyan Di, Sheeba Moghal, Bella Shi and Jacky Zhang"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# AutoAPI configuration
autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
]
autoapi_ignore = ["*/tests/*"]  # Ignore test files, if necessary

# List of patterns to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"