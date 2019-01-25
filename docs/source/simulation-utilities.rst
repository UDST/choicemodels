Simulation utilities API
========================

ChoiceModels provides general-purpose tools for Monte Carlo simulation of choices among alternatives, given probability distributions generated from fitted models. 

``monte_carlo_choices()`` is equivalent to applying ``np.random.choice()`` in parallel for many independent choice scenarios, but it's implemented as a single-pass matrix calculation that is much faster.

``iterative_lottery_choices()`` is for cases where the alternatives have limited capacitiesxs, requiring multiple passes to match choosers and alternatives. Effectively, choices are simulated sequentially, each time removing the chosen alternative or reducing its available capacity. (It's actually done in batches for better performance.)


Independent choices
-------------------

.. autofunction:: choicemodels.tools.monte_carlo_choices


Capacity-constrained choices
----------------------------

.. autofunction:: choicemodels.tools.iterative_lottery_choices