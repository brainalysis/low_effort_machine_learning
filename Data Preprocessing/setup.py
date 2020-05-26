# Copyright (C) 2019-2020 Fahad Akbar <fahad.akbar@gmail.com>
# License: MIT, m.akbar@queensu.ca , fahad.akbar@gmail.com

"""
PreProcess1 - An end-to-end open source data preprocessing tool

"""

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        README = f.read()
    return README


requirements = ["scikit-learn","datetime","datefinder","pyod","lightgbm"] # add plotly later on


setup(
    name="preprocess1",
    version="0.1.37",
    author="Fahad Akbar",
    author_email="fahad.akbar@gmail.com",
    description="A package to provide preprocessing steps for Machine Learning in an super easy way !",
    #long_description=readme,
    #long_description_content_type="text/markdown",
    url="https://github.com/mfahadakbar",
    packages=find_packages(),
    install_requires=requirements,
    #classifiers=[
    #    "Programming Language :: Python :: 3.7",
    #    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    #],
)
