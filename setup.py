from setuptools import setup

# read README as the long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='choicemodels',
    version='0.2.2',
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: BSD License'
    ],
    packages=['choicemodels', 'choicemodels.tools'],
    install_requires=[
        'numpy >= 1.14',
        'pandas >= 0.23',
        'patsy >= 0.5',
        'pylogit >= 0.2.2',
        'scipy >= 1.0',
        'statsmodels >= 0.8, <0.11; python_version <"3.6"',
        'statsmodels >= 0.8; python_version >="3.6"'
    ]
)
