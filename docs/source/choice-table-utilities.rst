Choice table utilities API
==========================

Working with discrete choice models can require a lot of data preparation. Each chooser has to be matched with hypothetical alternatives, either to simulate choice probabilities or to compare them with the chosen alternative for model estimation. 

ChoiceModels includes a class called ``MergedChoiceTable`` that automates this. To build a merged table, create an instance of the class and pass it one ``pd.DataFrame`` of choosers and another of alternatives, with whatever other arguments are needed (see below for full API). 

The merged data table can be output to a DataFrame, or passed directly to other ChoiceModels tools as a ``MergedChoiceTable`` object. (This retains metadata about indexes and other special columns.)

.. code-block:: python
   
   mct = choicemodels.tools.MergedChoiceTable(obs, alts, ..)
   df = mct.to_frame()

This tool is designed especially for models that need to sample from large numbers of alternatives. It supports:

- uniform random sampling of alternatives, with or without replacement
- weighted random sampling based either on characteristics of the alternatives or on combinations of chooser and alternative
- interaction terms to be merged onto the final data table
- cartesian merging of all the choosers with all the alternatives, without sampling

All of the sampling procedures work for both estimation (where the chosen alternative is known) and simulation (where it is not).


MergedChoiceTable
-----------------

.. autoclass:: choicemodels.tools.MergedChoiceTable
   :members: