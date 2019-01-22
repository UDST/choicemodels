[![Build Status](https://travis-ci.org/UDST/choicemodels.svg?branch=master)](https://travis-ci.org/UDST/choicemodels)
[![Coverage Status](https://coveralls.io/repos/github/UDST/choicemodels/badge.svg?branch=master)](https://coveralls.io/github/UDST/choicemodels?branch=master)

# ChoiceModels

This is a package for discrete choice model estimation and simulation, with an emphasis on large choice sets and behavioral refinements to multinomial models. Most of these models are not available in Statsmodels or Scikit-learn.

The underlying estimation routines come from two main places: (1) the `urbanchoice` codebase, which has been moved into ChoiceModels, and (2) Timothy Brathwaite's PyLogit package, which handles more flexible model specifications.



## Documentation

Package documentation is available on [readthedocs](https://choicemodels.readthedocs.io/).



## Installation

### Production releases

Production releases of ChoiceModels can be installed with pip or conda:

```
pip install choicemodels
```

```
conda install --channel conda-forge choicemodels
```

### Development releases

The latest development release can be installed using the Github URL. You may want to remove prior installations first to avoid version conflicts.

```
pip list
pip uninstall choicemodels
pip install git+git://github.com/udst/choicemodels.git
```

### Cloning the repository

If you will be editing ChoiceModels code or frequently updating to newer development versions, you can clone the repository and link it to your Python environment:

```
git clone https://github.com/udst/choicemodels.git
cd choicemodels
python setup.py develop
```

## Current functionality

`choicemodels.tools.MergedChoiceTable()`

- Generates a merged long-format table of choosers and alternatives.

`choicemodels.MultinomialLogit()`

- Fits MNL models, using either the ChoiceModels or PyLogit estimation engines.

`chociemodels.MultinomialLogitResults()`

- Stores and reports fitted MNL models.

There's documentation in these classes' docstrings, and a usage demo in a Jupyter notebook.

https://github.com/udst/choicemodels/blob/master/notebooks/Destination-choice-models-02.ipynb
