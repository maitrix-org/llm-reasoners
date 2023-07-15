#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='reasoners',
      version='0.0.0',
      packages=find_packages(exclude=('examples', 'examples.*')),
      install_requires=['tqdm',
                        'fire',
                        'numpy',
                        'scipy',
                        'torch',
                        'datasets',
                        'transformers',
                        'sentencepiece',
                        'llama@git+https://github.com/facebookresearch/llama',
                        'fairscale'])
