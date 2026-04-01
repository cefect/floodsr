"""Sphinx configuration for floodsr documentation."""

import os
from pathlib import Path
from datetime import datetime
from setuptools_scm import get_version

# -- Project information -----------------------------------------------------

project = "floodsr"
author = "floodsr developers"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# Derive the docs version from SCM tags so RTD stays aligned with releases.
release = get_version(root="../..", relative_to=__file__)
version = release

# -- General configuration ---------------------------------------------------

# Core extensions kept intentionally small for an MVP docs site.
extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "myst_nb",
    "sphinx_copybutton",
]

# Prefix section labels with document path to avoid collisions as docs grow.
autosectionlabel_prefix_document = True

# Keep templates and static assets local to docs/.
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "readme.md", "notebooks/readme.md"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Keep notebook examples source-only in docs builds.
nb_execution_mode = "off"


# Normalize RTD language slugs so Sphinx can find locale catalogs.
language = "fr" if (os.environ.get("READTHEDOCS_LANGUAGE") or "en") == "fr" else "en"
assert language != "fr" or (Path(__file__).resolve().parent / "locale" / "fr" / "LC_MESSAGES" / "index.po").is_file(), "French docs require docs/user/locale/fr/LC_MESSAGES/index.po; RTD would otherwise fall back to English."
locale_dirs = ["locale/"]
gettext_compact = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "floodsr docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/cefect/floodsr",
    "repository_branch": "master",
    "path_to_docs": "docs/user",
    "use_download_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
    },
}
html_context = {
    "default_mode": "light",
}
