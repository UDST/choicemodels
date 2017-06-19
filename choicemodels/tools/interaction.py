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
	
	Parameters
	----------
	choosers : pandas.DataFrame
		Table with one row for each chooser, with unique ID's in the Index field.
	alternatives : pandas.DataFrame
		Table with one row for each alternative, with unique ID's in the Index field.
	chosen_alternatives : str or list or pandas.Series, optional
		List of the alternative ID selected by each chooser. If str, interpreted as a 
		column of the choosers table. If list, must be in the same order as the choosers.
	sample_size : int, optional
		Number of alternatives available for each chooser. These will be sampled randomly.
		If 'None', all of the alternatives will be available for each chooser.
	sample_weights : ???, optional
		Not yet implemented. The weights could either be alternative-specific, 
		(len = alts) or could vary for different choosers (len = alts x choosers). So we
		might want to accept either (a) a column of the alternatives table, (b) a separate
		lookup table, or (c) a function that returns weights, similar to what's done
		with lookups elsewhere in UrbanSim...
	
	Properties
	----------
	chosen : 2D matrix
		Representation of the chosen alternatives that aligns with the merged data table.
		For now, this is a binary matrix, but a column might work better with PyLogit.
	sampled_alternatives : pandas.Series
		ID's of the alternatives that were sampled. Not sure we need this, but I'm 
		including it for backwards compatibility with urbansim.urbanchoice.
	
	"""
	def __init__(self, choosers, alternatives, chosen_alternatives=None, 
				 sample_size=None, sample_weights=None):
		
		# TO DO: implement case where sample_size = None
		
		alts, merged, chosen = mnl_interaction_dataset(choosers, alternatives,
													   sample_size, chosen_alternatives)
		self._merged_table = merged
		self.sampled_alternatives = alts
		self.chosen = chosen
		return

	def to_frame(self):
		"""
		Long-format DataFrame of the merged table. The alternatives for a particular 
		chooser are contiguous, with the chosen alternative listed first. 
		
		[TO DO: Add an explicit indication of the chosen alternative?]
		
		"""
		return self._merged_table


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

    alts_sample = alternatives.take(sample)
    assert len(alts_sample.index) == SAMPLE_SIZE * len(choosers.index)
    alts_sample['join_index'] = np.repeat(choosers.index.values, SAMPLE_SIZE)

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='join_index', right_index=True,
        suffixes=('', '_r'))

    chosen = np.zeros((numchoosers, SAMPLE_SIZE))
    chosen[:, 0] = 1

    logger.debug('finish: compute MNL interaction dataset')
    return alternatives.index.values[sample], alts_sample, chosen
