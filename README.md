# ChoiceModels

ChoiceModels is a Python library for discrete choice modeling, with utilities for sampling, simulation, and other ancillary tasks. It's part of the [Urban Data Science Toolkit](https://docs.udst.org) (UDST).


### Features

The library focuses mainly on tools to help integrate discrete choice models into larger workflows, drawing on other packages such as the excellent [PyLogit](https://github.com/timothyb0912/pylogit) for most estimation of models. 

ChoiceModels can automate the creation of choice tables for estimation or simulation, using uniform or weighted random sampling of alternatives, as well as interaction terms or cartesian merges. 

It also provides general-purpose tools for Monte Carlo simulation of choices given probability distributions from fitted models, with fast algorithms for independent or capacity-constrained choices. 

ChoiceModels includes a custom engine for Multinomial Logit estimation that's optimized for fast performance with large numbers of alternatives.


### Installation

ChoiceModels can be installed using the Pip or Conda package managers:

```
pip install choicemodels
```

```
conda install choicemodels --channel conda-forge
```


### Documentation

See the online documentation for much more: https://udst.github.io/choicemodels

Some additional documentation is available within the repo in `CHANGELOG.md`, `CONTRIBUTING.md`, `/docs/README.md`, and `/tests/README.md`.

There's discussion of current and planned features in the [Pull requests](https://github.com/udst/choicemodels/pulls?utf8=✓&q=is%3Apr) and [Issues](https://github.com/udst/choicemodels/issues?utf8=✓&q=is%3Aissue), both open and closed.
