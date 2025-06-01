# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Add this near the top of the file, after the imports

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Go up two levels to reach project root

# Add a quick test to verify imports are working
try:
    import Baryonic_Correction
    print(f"BCM imported successfully from {Baryonic_Correction.__file__}")
except ImportError as e:
    print(f"Failed to import BCM: {e}")

project = 'Baryonic_Correction_Model'
copyright = '2025, Max Rauter'
author = 'Max Rauter'
release = '1.0.0-alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = []

# Mock imports for packages that might cause installation issues on RTD
autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "h5py", "hdf5plugin", "tqdm"]

# Autodoc settings
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Napoleon settings for better docstring parsing
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_title = "Baryonic Correction Model"
#html_logo = "_static/logo.png"  # Add your logo
#html_favicon = "_static/favicon.ico"

# Custom CSS
html_css_files = [
    'custom.css',
]