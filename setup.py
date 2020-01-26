from setuptools import setup, find_packages

requirements = ["scikit-learn","datetime","datefinder"] # add plotly later on

# this is where you mention dependencies for your package
# only add libraries that are not by default installed by python , so if your package only
# used say pandas or numpy , you dont need to put them here as they are already installed 

setup(
    name="preprocess1",
    version="0.1.27",
    author="Fahad Akbar",
    author_email="fahadakbar@gmail.com",
    description="A package to provide easy preprocessing steps for Machine Learning in an easy way",
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
