[![Build Status](https://travis-ci.org/UDST/choicemodels.svg?branch=master)](https://travis-ci.org/UDST/choicemodels)
[![Coverage Status](https://coveralls.io/repos/github/UDST/choicemodels/badge.svg?branch=master)](https://coveralls.io/github/UDST/choicemodels?branch=master)

# ChoiceModels

This is a package for discrete choice model estimation and simulation, with an emphasis on large choice sets and behavioral refinements to multinomial models. Most of these models are not available in Statsmodels or Scikit-learn.

The underlying estimation routines come from two main places: (1) the `urbanchoice` codebase, which has been moved into ChoiceModels, and (2) Timothy Brathwaite's PyLogit package, which handles more flexible model specifications.



## Documentation

Package documentation is available on [readthedocs](https://choicemodels.readthedocs.io/).



## Installation

Install with pip:

`pip install choicemodels`

or with conda-forge.



## Current functionality

`choicemodels.tools.MergedChoiceTable()`

- Generates a merged long-format table of choosers and alternatives.

`choicemodels.MultinomialLogit()`

- Fits MNL models, using either the ChoiceModels or PyLogit estimation engines.

`chociemodels.MultinomialLogitResults()`

- Stores and reports fitted MNL models.

There's documentation in these classes' docstrings, and a usage demo in a Jupyter notebook.

https://github.com/udst/choicemodels/blob/master/notebooks/Destination-choice-models-02.ipynb
