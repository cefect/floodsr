"""Sphinx configuration for floodsr documentation."""

from datetime import datetime

# -- Project information -----------------------------------------------------

project = "floodsr"
author = "floodsr developers"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"
release = "0.0.1"
version = release

# -- General configuration ---------------------------------------------------

# Core extensions kept intentionally small for an MVP docs site.
extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Prefix section labels with document path to avoid collisions as docs grow.
autosectionlabel_prefix_document = True

# Keep templates and static assets local to docs/.
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "readme.md"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "floodsr docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
