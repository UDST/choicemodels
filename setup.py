# Install setuptools if not installed.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages


# read README as the long description
with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = f.readlines()
install_requires = [item.strip() for item in install_requires]

setup(
    name='choicemodels',
    version='0.1dev',
    description='Tools for discrete choice estimation',
    long_description=long_description,
    author='Urban Analytics Lab',
    url='https://github.com/ual/choicemodels',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=install_requires
)
