from setuptools import setup, find_packages

requirements = ["scikit-learn","datetime","datefinder","pyod"] # add plotly later on


setup(
    name="preprocess1",
    version="0.1.34",
    author="Fahad Akbar",
    author_email="fahadakbar@gmail.com",
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
