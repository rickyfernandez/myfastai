#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = """
    torch>=1.2.0 torchvision matplotlib pandas requests pyyaml fastprogress pillow scikit-learn scipy spacy
""".split()

setup_requirements = ['setuptools>=36.2']

setup(
    name = 'learnai',
    packages = find_packages(),
    include_package_data = True,

    install_requires = requirements,
    setup_requires   = setup_requirements,
    python_requires  = '>=3.6',

    description = "fastai v2",
    long_description_content_type = 'text/markdown',

    zip_safe = False,
)
