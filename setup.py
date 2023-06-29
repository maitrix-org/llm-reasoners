#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='rap',
      version='0.0.0',
      packages=find_packages(exclude=('examples', 'examples.*')))
