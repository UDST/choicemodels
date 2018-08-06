"""
Utilities for making merged interaction tables of choosers and
the alternatives from which they are choosing.
Used for location choice models.

"""
import logging

import numpy as np
import pandas as pd

from . import pmat

"""
########################
NEW FUNCTION DEFINITIONS
########################

I'm trying setting this up as a class instead of a function. Generally I'm not a fan of
functions that return multiple items. If MergedChoiceTable() is a class, then we can
provide access to different properties as needed..

The original function returns three things:
- ID's of the alternatives that were sampled. I don't think this is necessary, because
  you can easily determine it from the merged dataset, right?
- Merged dataset in long format. This is the key output.
- 2D binary matrix representing chosen alternatives. This is the format expected by the
  MNL estimation function, but it could also be provided as a named column in the merged
  dataset.

There may be other properties we'll want for PyLogit compatibility, too.

"""

class MergedChoiceTable(object):
    """
    Generates a merged long-format table of choosers and alternatives, for discrete choice
    model estimation or simulation. 
    
    Attributes that vary based on interaction between the choosers and alternatives 
    (distance, for example) will need to be added in post-processing. 
    
    Reserved column names: 'chosen', 'join_index', 'observation_id'.
    
    Parameters
    ----------
    observations : pandas.DataFrame
        Table with one row for each choice scenario, with unique ID's in the index field.
        Additional columns can contain fixed attributes of the choosers. Best not to have
        a column with the same name as the index field of the 'alternatives' table,
        because these will clash when the tables are merged. (If you have one, it will be
        used to identify chosen alternatives if needed, and then dropped.)
        [TO DO: check that this is handled correctly.]

    alternatives : pandas.DataFrame
        Table with one row for each alternative, with unique ID's in the index field.
        Additional columns can contain fixed attributes of the alternatives.

    chosen_alternatives : str or list or pandas.Series, optional
        List of the alternative ID selected in each choice scenario. (This is required for
        preparing estimation data, but not for simulation data.) If str, interpreted as a
        column name from the observations table. If list, must be in the same order as the
        observations. The column will be dropped from the merged table and replaced with a
        binary column named 'chosen'.

    sample_size : int, optional
        Number of alternatives available for each chooser. These will be sampled randomly.
        If 'None', all of the alternatives will be available for each chooser.

    weights : str, pandas.Series, or callable, optional
        Weights to apply when sampling alternatives. (A) One weight per alternative. If 
        str, interpreted as a column name from the alternatives table. If Series, it 
        should have the same length as the alternatives table and its index should align.
        (B) One weight per combination of observation and alternative. If Series, it
        should have one row per combination of observation id and alternative id, with
        the first index level corresponding to the former and the second index level
        to the latter. If callable, it should accept two arguments in the form of 
        `observation_id`, `alternative_id` and return the corresponding weight.

    """
    def __init__(self, observations, alternatives, chosen_alternatives=None,
                 sample_size=None, weights=None):

        # TO DO: implement case where sample_size = None
        # TO DO: implement case where chosen_alternatives is a string (might be nice to
        #        drop it from the merged data table, to keep things clean?)
        # TO DO: implement case where chosen_alternatives = None

        alts, merged, chosen = mnl_interaction_dataset(observations, alternatives,
                                                       sample_size, chosen_alternatives)

        # Convert the representation of chosen alternatives to a column in table
        merged['chosen'] = np.reshape(chosen.astype(int), (merged.shape[0], 1))

        # Label the observation id  [TO DO: would be nice to keep original name]
        merged = merged.rename(columns = {'join_index': 'observation_id'})

        # Store the alternative id
        self._alternative_id_col = merged.index.name

        if (self._alternative_id_col in merged.columns):
            # if there's an existing column with same name, drop it
            merged = merged.drop(self._alternative_id_col, axis=1)

        merged = merged.reset_index()  # save as column

        self._merged_table = merged
        return

    def to_frame(self):
        """
        Long-format DataFrame of the merged table. The rows representing alternatives for
        a particular chooser are contiguous, with the chosen alternative listed first.

        """
        return self._merged_table

    @property
    def observation_id_col(self):
        """
        Name of column in the merged table containing the observation id. Values will
        match the index of the 'choosers' table, but for now the column name is reset.

        """
        return 'observation_id'

    @property
    def alternative_id_col(self):
        """
        Name of column in the merged table containing the alternative id. Name and values
        will match the index of the 'alternatives' table,

        """
        return self._alternative_id_col

    @property
    def choice_col(self):
        """
        Name of the generated column containing a binary representation of whether each
        alternative was chosen in the given choice scenario.

        """
        return 'chosen'


"""
#############################
ORIGINAL FUNCTION DEFINITIONS
#############################

"""

logger = logging.getLogger(__name__)
GPU = False


def enable_gpu():
    global GPU
    GPU = 1
    pmat.initialize_gpu()


# TODO: split this out into separate functions for estimation
# and simulation.
def mnl_interaction_dataset(choosers, alternatives, SAMPLE_SIZE,
                            chosenalts=None):
    logger.debug((
        'start: compute MNL interaction dataset with {} choosers, '
        '{} alternatives, and sample_size={}'
        ).format(len(choosers), len(alternatives), SAMPLE_SIZE))
    # filter choosers and their current choices if they point to
    # something that isn't in the alternatives table
    if chosenalts is not None:
        isin = chosenalts.isin(alternatives.index)
        try:
            removing = isin.value_counts().loc[False]
        except Exception:
            removing = None
        if removing:
            logger.info((
                "Removing {} choice situations because chosen "
                "alt doesn't exist"
            ).format(removing))
            choosers = choosers[isin]
            chosenalts = chosenalts[isin]

    numchoosers = choosers.shape[0]
    numalts = alternatives.shape[0]

    # TODO: this is currently broken in a situation where
    # SAMPLE_SIZE >= numalts. That may not happen often in
    # practical situations but it should be supported
    # because a) why not? and b) testing.
    alts_idx = np.arange(len(alternatives))
    if SAMPLE_SIZE < numalts:
        # TODO: Use stdlib random.sample to individually choose
        # alternatives for each chooser (to avoid repeatedly choosing the
        # same alternative).
        # random.sample is much faster than np.random.choice.
        # [Also, it looks like this could potentially sample the same alternative more
        #  than once for a single choice scenario, which is not ideal. -Sam]
        sample = np.random.choice(alts_idx, SAMPLE_SIZE * numchoosers)
        if chosenalts is not None:
            # replace the first row for each chooser with
            # the currently chosen alternative.
            # chosenalts -> integer position
            sample[::SAMPLE_SIZE] = pd.Series(
                alts_idx, index=alternatives.index).loc[chosenalts].values
    else:
        assert chosenalts is None  # if not sampling, must be simulating
        sample = np.tile(alts_idx, numchoosers)

    if not choosers.index.is_unique:
        raise Exception(
            "ERROR: choosers index is not unique, "
            "sample will not work correctly")
    if not alternatives.index.is_unique:
        raise Exception(
            "ERROR: alternatives index is not unique, "
            "sample will not work correctly")

    alts_sample = alternatives.take(sample).copy()
    assert len(alts_sample.index) == SAMPLE_SIZE * len(choosers.index)
    alts_sample['join_index'] = np.repeat(choosers.index.values, SAMPLE_SIZE)

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='join_index', right_index=True,
        suffixes=('', '_r'))

    chosen = np.zeros((numchoosers, SAMPLE_SIZE))
    chosen[:, 0] = 1

    logger.debug('finish: compute MNL interaction dataset')
    return alternatives.index.values[sample], alts_sample, chosen
