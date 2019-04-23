Multinomial Logit API
=====================

ChoiceModels has built-in functionality for Multinomial Logit estimation and simulation. This can use either the `PyLogit <https://github.com/timothyb0912/pylogit>`__ MNL estimation engine or a custom engine optimized for fast performance with large numbers of alternatives. The custom engine is originally from ``urbansim.urbanchoice``.

Fitting a model yields a results object that can generate choice probabilities for out-of-sample scenarios.


MultinomialLogit
----------------

.. autoclass:: choicemodels.MultinomialLogit
   :members:


MultinomialLogitResults
-----------------------

.. autoclass:: choicemodels.MultinomialLogitResults
   :members:
