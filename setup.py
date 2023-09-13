#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='reasoners',
      version='0.0.0',
      packages=['exllama', 'reasoners'], #  find_packages(exclude=('examples', 'examples.*')),
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
                        'llama1@git+https://github.com/AegeanYan/llama@llama_v1',  # llama 2 have some alias problem so you may need to clone the forked llama1 at private repo. please check the setup.py
                        'llama@git+https://github.com/facebookresearch/llama@main',
                        'fairscale'],
      include_package_data=True,
      extras_require={
          'awq': ['awq@git+https://github.com/mit-han-lab/llm-awq',
                  'awq_inference_engine@git+https://github.com/mit-han-lab/llm-awq.git@main#subdirectory=awq/kernels'],
      },
      python_requires='>=3.10')
