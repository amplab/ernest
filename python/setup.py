#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
	name='ernest',
	maintainer='Shivaram Venkataraman',
	maintainer_email='shivaram@cs.berkeley.edu',
	version='0.1',
	description='Performance Prediction for Spark jobs.',
	long_description=open('../README.md').read(),
	url='https://github.com/amplab/ernest',
	license='Apache License 2.0',

	packages=find_packages(),
	include_package_data=True,

  install_requires=["cvxpy >= 0.2.22",
                    "numpy >= 1.8",
                    "scipy >= 0.13"],
)
