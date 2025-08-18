from setuptools import setup, find_packages
from ralf.__init__ import __version__

setup(
    name="ralf",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "pandas",
        "kagglehub",
        "peft",
        "transformers",
        "scikit-learn",
        "datasets",
        "evaluate",
    ],
    author="Venkatesh Tadinada",
    description="RALF is a data-driven framework for selecting the most cost-effective, secure, and accurate LLM for any enterprise task",
    python_requires=">=3.8",
)