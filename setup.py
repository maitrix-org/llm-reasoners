#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='reasoners',
      version='0.0.0',
      packages=['exllama', 'reasoners'],  # find_packages(exclude=('examples', 'examples.*')),
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
                        'llama1@git+https://github.com/AegeanYan/llama@llama_v1',  # To avoid the alias problem, you need to clone the forked llama1
                        'llama@git+https://github.com/facebookresearch/llama@main',  # llama 2
                        'llama3@git+https://github.com/Ber666/llama3@llama3',
                        'fairscale',
                        'google-generativeai',
                        'anthropic'],
      include_package_data=True,
      extras_require={
          'awq': ['awq@git+https://github.com/mit-han-lab/llm-awq',
                  'awq_inference_engine@git+https://github.com/mit-han-lab/llm-awq.git@main#subdirectory=awq/kernels'],
      },
      python_requires='>=3.10')
