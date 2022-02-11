from setuptools import setup, find_packages
from pathlib import Path
import re

this_directory = Path(__file__).parent
#long_description = (this_directory / "README.md").read_text()



setup(
    name='optim',
    version="0.0.1",
    description="Optimization methods",
    long_description="Optimization methods",
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
    ],
    packages=find_packages(),
    scripts=[],
    #tests_require=['pytest'],
    author='Alexandre Schulz',
    author_email='',
    url='',
    license='MIT',
)
