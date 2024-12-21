# thisi s depricated see blow line of comments (line 22)
from setuptools import setup, find_packages
setup(
    name="fast", 
    version="1.0", 
    author="Diego Ligtenberg",
    author_email="diegoligtenberg@gmail.com",
    description="Foundational Audio Spectrogram Transformer-based model",
    url = None,
    license="MIT",
    long_description=("README"),
    packages=find_packages(where="src"),
    package_dir={"": "src"}
    )

# print(find_packages())
# can install this with:     pip install -e .
# can uninstall this with:   pip uninstall mss_project

#TODO add ffmpeg as requirement

# this is depricated, we now need to use pyproject.toml
# pip install -e . --use-pep517 # in the pyproject.tml file we specify the package name 'fast'
# pip uninstall fast