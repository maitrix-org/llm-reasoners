#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='llm-reasoners',
      version='1.0.2',
      description='A library for advanced reasoning methods with large language models',
      packages=find_packages(include=['reasoners', 'reasoners.*']),
      # packages=find_packages(exclude=('examples', 'examples.*')),
      entry_points={
          'console_scripts': ['reasoners-visualizer=reasoners.visualization:main'],
      },
      install_requires=['tqdm',
                        'fire',
                        'numpy',
                        'scipy',
                        'torch',
                        'datasets',
                        'huggingface_hub',
                        'transformers',
                        'sentencepiece',
                        'openai',
                        'tarski',
                        'peft',
                        'optimum',
                        'ninja',
                        'bitsandbytes',
                        'fairscale',
                        'google-generativeai',
                        'anthropic'],
      include_package_data=True,
      python_requires='>=3.10')