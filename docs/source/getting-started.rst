Getting started
===============

Intro
-----

ChoiceModels is a Python library for discrete choice modeling, with utilities for sampling, simulation, and other ancillary tasks. It's part of the `Urban Data Science Toolkit <https://docs.udst.org>`__ (UDST).

The library focuses mainly on tools to help integrate discrete choice models into larger workflows, drawing on other packages such as the excellent `PyLogit <https://github.com/timothyb0912/pylogit>`__ for most estimation of models. ChoiceModels can automate the creation of choice tables for estimation or simulation, using uniform or weighted random sampling of alternatives, as well as interaction terms or cartesian merges. It also provides general-purpose tools for Monte Carlo simulation of choices given probability distributions from fitted models, with fast algorithms for independent or capacity-constrained choices. ChoiceModels includes a custom engine for Multinomial Logit estimation that's optimized for fast performance with large numbers of alternatives.

ChoiceModels is `hosted on Github <https://github.com/udst/choicemodels>`__ with a BSD 3-Clause open source license. The code repository includes some material not found in this documentation: a `change log <https://github.com/UDST/choicemodels/blob/main/CHANGELOG.md>`__, a `contributor's guide <https://github.com/UDST/choicemodels/blob/main/CONTRIBUTING.md>`__, and instructions for `running the tests <https://github.com/UDST/choicemodels/tree/main/tests>`__, `updating the documentation <https://github.com/UDST/choicemodels/tree/main/docs>`__, and `creating a new release <https://github.com/UDST/choicemodels/blob/main/CONTRIBUTING.md>`__. Another useful resource is the `issues <https://github.com/UDST/choicemodels/issues?utf8=✓&q=is%3Aissue>`__ and `pull requests <https://github.com/UDST/choicemodels/pulls?q=is%3Apr>`__ on Github, which include detailed feature proposals and other discussions.

ChoiceModels was created in 2016, with contributions from Sam Maurer (maurer@urbansim.com), Timothy Brathwaite, Geoff Boeing, Paul Waddell, Max Gardner, Eddie Janowicz, Arezoo Besharati Zadeh, Jacob Finkelman, Catalina Vanoli, and others. It includes earlier code written by Matt Davis, Fletcher Foti, and Paul Sohn.


Installation
------------

ChoiceModels is tested with Python 2.7, 3.5, 3.6, 3.7, and 3.8. It should run on any platform. 


Production releases
~~~~~~~~~~~~~~~~~~~

ChoiceModels can be installed using the Pip or Conda package managers. We recommend Conda because it resolves dependency conflicts better.

.. code-block:: python

    pip install choicemodels

.. code-block:: python

    conda install choicemodels --channel conda-forge


When new production releases of ChoiceModels come out, you can upgrade like this:

.. code-block:: python

    pip install choicemodels --upgrade

.. code-block:: python

    conda update choicemodels --channel conda-forge


Developer pre-releases
~~~~~~~~~~~~~~~~~~~~~~

Developer pre-releases of ChoiceModels can be installed using the Github URL. Additional information about the developer releases can be found in Github `pull requests <https://github.com/UDST/choicemodels/pulls?q=is%3Apr>`__.

.. code-block:: python

    pip install git+git://github.com/udst/choicemodels.git

You can use the same command to upgrade.


Cloning the repository
~~~~~~~~~~~~~~~~~~~~~~

You can also install ChoiceModels by cloning the Github repository, which is the best way to do it if you'll be modifying the code. The main branch contains the latest developer release. 

.. code-block:: python

    git clone https://github.com/udst/choicemodels.git
    cd choicemodels
    python setup.py develop

Update it with ``git pull``.


Basic usage
-----------

You can use components of ChoiceModels individually, or combine them together to streamline model estimation and simulation workflows. Other UDST libraries like UrbanSim Templates use ChoiceModels objects as inputs and outputs.

If you have choosers and alternatives as Pandas DataFrames, you can prepare them for model estimation like this:

.. code-block:: python
   
   mct = choicemodels.tools.MergedChoiceTable(obs, alts, chosen_alternatives='chosen',
                                              sample_size=10, ..)

Then, you can estimate a Multinomial Logit model like this:

.. code-block:: python
   
   results = choicemodels.MultinomialLogit(mct, model_expression='x1 + x2 + x3')

This provides a ``choicemodels.MultinomialLogitResults`` object, from which you can obtain probability distributions for out-of-sample choice scenarios in order to generate simulated choices.

.. code-block:: python
   
   mct2 = choicemodels.tools.MergedChoiceTable(obs2, alts, sample_size=10, ..)
   probs = results.probabilities(mct2)
   choices = choicemodels.tools.monte_carlo_choices(probs)


