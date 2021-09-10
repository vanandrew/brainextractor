#!/usr/bin/env python3
from distutils.core import setup

setup(
    name='BrainExtractor',
    version='0.0.0',
    description='A re-implementation of FSL\'s Brain Extraction Tool in Python',
    author='Andrew Van',
    author_email='vanandrew@wustl.edu',
    packages=['brainextractor'],
    install_requires=[
        'trimesh>=3.8.15',
        'pyrender>=0.1.43',
        'numpy>=1.19.4',
        'scipy>=1.5.4',
        'numba>=0.51.2',
        'nibabel>=3.2.1'
    ],
    scripts=[
        'scripts/brainextractor',
        'scripts/brainextractor_render'
    ]
)
