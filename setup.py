# Install setuptools if not installed.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup


# read README as the long description
with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = f.readlines()
install_requires = [item.strip() for item in install_requires]

setup(
    name='choicemodels',
    version='0.1',
    description='Tools for discrete choice estimation',
    long_description=long_description,
    author='UDST',
    url='https://github.com/udst/choicemodels',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: BSD License'
    ],
    packages=['choicemodels'],
    install_requires=install_requires
)
