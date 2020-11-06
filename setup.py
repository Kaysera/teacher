from setuptools import setup, find_packages

install_requires = [
    'pandas',
    'scikit-fuzzy',
    'scikit-learn',
    'matplotlib',
    'deap',
    'imblearn'
]

setup(name="flore", packages=find_packages())
