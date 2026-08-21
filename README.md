# ChoiceModels

[![CI](https://github.com/UDST/choicemodels/actions/workflows/ci.yml/badge.svg)](https://github.com/UDST/choicemodels/actions/workflows/ci.yml)

ChoiceModels is a Python library for discrete choice modeling, with utilities for sampling, simulation, and other ancillary tasks. It's part of the [Urban Data Science Toolkit](https://docs.udst.org) (UDST).

## Project scope

**Status:** Active

**Mission:** ChoiceModels provides reusable tools for specifying, estimating,
sampling, and simulating discrete-choice models within larger analytical
workflows.

**Architecture:** ChoiceModels provides portable Python implementations and
interfaces for discrete-choice workflows and conventional estimation. Its
reference implementations target conventional CPU-based execution, while
estimator interfaces allow alternative execution engines to interoperate with
the library.

The project maintains and develops:

- construction and manipulation of choice tables;
- sampling of alternatives;
- discrete-choice model specifications and interfaces;
- conventional model estimation and estimator integrations;
- Monte Carlo choice simulation;
- capacity-constrained choice algorithms; and
- interoperable representations of estimated choice models.

ChoiceModels is designed as a general-purpose discrete-choice library and as
an integration layer between model specifications, estimation engines, and
simulation workflows.

Development of choice-model methods and estimator integrations is welcome
within this mission and architecture. Material changes to the project's
mission or execution architecture are considered through UDST's
organization-level governance process.

See the [UDST Project Directory](https://github.com/UDST/.github/blob/main/PROJECTS.md)
for organization-wide project status and policy.


### Features

The library focuses mainly on tools to help integrate discrete choice models into larger workflows, drawing on other packages such as the excellent [PyLogit](https://github.com/timothyb0912/pylogit) for most estimation of models. 

ChoiceModels can automate the creation of choice tables for estimation or simulation, using uniform or weighted random sampling of alternatives, as well as interaction terms or cartesian merges. 

It also provides general-purpose tools for Monte Carlo simulation of choices given probability distributions from fitted models, with fast algorithms for independent or capacity-constrained choices. 

ChoiceModels includes a custom engine for Multinomial Logit estimation that's optimized for fast performance with large numbers of alternatives.


### Installation

ChoiceModels requires Python 3.10 or later, and can be installed using the Pip or Conda package managers:

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
