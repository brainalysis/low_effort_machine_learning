from setuptools import setup, find_packages

requirements = ["ipython>=6"] 
# this is where you mention dependencies for your package
# only add libraries that are not by default installed by python , so if your package only
# used say pandas or numpy , you dont need to put them here as they are already installed 

setup(
    name="preprocess",
    version="0.0.1",
    author="Fahad Akbar",
    author_email="fahadakbar@gmail.com",
    description="A package to eliminate multicollinearity",
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
