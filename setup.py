from setuptools import setup, find_packages
import sys, os.path

setup(name='Salford Robotics Gym',
      author="xiaochen",
      python_requires='>=3.5',
      packages=[package for package in find_packages() if package.startswith('srg')])