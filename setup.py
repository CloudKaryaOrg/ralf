from setuptools import setup, find_packages
# from .ralf.__init__ import __version__

import os

# Read the version from my_package/__init__.py
def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'ralf', '__init__.py')) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    raise RuntimeError("Unable to find __version__ string.")

setup(
    name="ralf",
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "pandas",
        "kagglehub",
        "transformers",
        "scikit-learn",
        "datasets",
        "evaluate",
    ],
    author="Venkatesh Tadinada",
    description="RALF is a data-driven framework for selecting the most cost-effective, secure, and accurate LLM for any enterprise task",
    python_requires=">=3.8",
)

#    version=__version__,
#        "peft",
