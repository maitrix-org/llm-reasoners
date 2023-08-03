#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='reasoners',
      version='0.0.0',
      packages=find_packages(exclude=('examples', 'examples.*')),
      entry_points={
          'console_scripts': ['reasoners-visualizer=reasoners.visualization:main'],
      },
      install_requires=['tqdm',
                        'fire',
                        'numpy',
                        'scipy',
                        'torch',
                        'datasets',
                        'transformers',
                        'sentencepiece',
                        'openai',
                        'llama1@git+https://github.com/AegeanYan/llama@llama_v1',
                        'llama@git+https://github.com/facebookresearch/llama@main',
                        'fairscale'])
