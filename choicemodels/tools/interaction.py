"""
Utilities for making merged interaction tables of choosers and
the alternatives from which they are choosing.
Used for location choice models.

"""
import logging
import random

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

class MCT(object):
    """
    Work in progress refactoring the choice table generation
    
    """

    def __init__(self, observations, alternatives, chosen_alternatives=None,
                 sample_size=None, replace=True, weights=None, random_state=None):
        
        # Validate the inputs...
        
        if isinstance(sample_size, float):
            sample_size = int(sample_size)
        
        if (sample_size is not None):
            if (sample_size <= 0):
                raise ValueError("Cannot sample {} alternatives; to run without sampling "
                        "leave sample_size=None".format(sample_size))

            if (replace == False) & (sample_size > alternatives.shape[0]):
                raise ValueError("Cannot sample without replacement with sample_size {} "
                        "and n_alts {}".format(sample_size, alternatives.shape[0]))
        
        # TO DO - check that dfs have unique indexes
        # TO DO - check that chosen_alternatives correspond correctly to other dfs
        # TO DO - same with weights (could join onto other tables and then split off)
        # TO DO - check for overlapping column names
        
        # Normalize chosen_alternatives to a pd.Series
        if (chosen_alternatives is not None) & isinstance(chosen_alternatives, str):
            chosen_alternatives = observations[chosen_alternatives]
            observations = observations.drop(chosen_alternatives.name, axis='columns')
        
        # Normalize weights to a pd.Series
        if (weights is not None) & isinstance(weights, str):
            weights = alternatives[weights]
        
        weights_1d = False
        weights_2d = False
        if (weights is not None):
            if (len(weights) == len(alternatives)):
                weights_1d = True
            elif (len(weights) == len(observations) * len(alternatives)):
                weights_2d = True
        
        self.observations = observations
        self.alternatives = alternatives
        self.chosen_alternatives = chosen_alternatives
        self.sample_size = sample_size
        self.replace = replace
        self.weights = weights
        self.random_state = random_state

        self.weights_1d = weights_1d
        self.weights_2d = weights_2d
        
        # Build choice table...
        
        if (sample_size is None):
            self._merged_table = self._build_table_without_sampling()
        
        elif (replace == True) & (weights_2d == False):
            self._merged_table = self._build_table_with_single_sample()
        
        else:
            # TO DO - make third condition explicit
            self._merged_table = self._build_table_with_repeated_sample()
        
        
    def _build_table_without_sampling(self):
        """
        This handles the cases where each alternative is available for each chooser.
        
        Expected class parameters
        -------------------------
        self.observations : pd.DataFrame
        self.alternatives : pd.DataFrame
        self.chosen_alternatives : pd.Series or None
        self.sample_size : None

        """
        oid_name = self.observations.index.name
        aid_name = self.alternatives.index.name
        
        obs_ids = np.repeat(self.observations.index.values, len(self.alternatives))
        alt_ids = np.tile(self.alternatives.index.values, reps=len(self.observations))
        
        df = pd.DataFrame({oid_name: obs_ids, aid_name: alt_ids})
   
        df = df.join(self.observations, how='left', on=oid_name)
        df = df.join(self.alternatives, how='left', on=aid_name)
        
        if (self.chosen_alternatives is not None):
            df['chosen'] = 0
            df = df.join(self.chosen_alternatives, how='left', on=oid_name)
            df.loc[df[aid_name] == df[self.chosen_alternatives.name], 'chosen'] = 1
            df.drop(self.chosen_alternatives.name, axis='columns', inplace=True)
        
        df.set_index([oid_name, aid_name], inplace=True)
        return df

    
    def _build_table_with_single_sample(self):
        """
        This handles the cases where we can draw a single sample and distribute it among
        the choosers -- sampling with replacement, with optional alternative-specific
        weights but NOT weights that apply to combinations of observation x alternative.
        
        Expected class parameters
        -------------------------
        self.observations : pd.DataFrame
        self.alternatives : pd.DataFrame
        self.chosen_alternatives : pd.Series or None
        self.sample_size : int
        self.replace : True
        self.weights : pd.Series or None
        self.random_state : NOT YET IMPLEMENTED

        """
        n_obs = self.observations.shape[0]
        
        oid_name = self.observations.index.name
        aid_name = self.alternatives.index.name
        
        samp_size = self.sample_size
        if (self.chosen_alternatives is not None):
            samp_size = self.sample_size - 1
        
        obs_ids = np.repeat(self.observations.index.values, samp_size)
        
        # No weights: core python is most efficient
        if (self.weights is None):
            alt_ids = random.choices(self.alternatives.index.values, 
                                     k = n_obs * samp_size)
        
        # Alternative-specific weights: numpy is most efficient
        else:
            alt_ids = np.random.choice(self.alternatives.index.values, 
                                       replace = True,
                                       p = self.weights/self.weights.sum(),
                                       size = n_obs * samp_size)
        
        chosen = np.repeat(0, samp_size * len(self.observations))
        if (self.chosen_alternatives is not None):
            obs_ids = np.append(obs_ids, self.observations.index.values)
            alt_ids = np.append(alt_ids, self.chosen_alternatives)
            chosen = np.append(chosen, np.repeat(1, len(self.observations)))
        
        df = pd.DataFrame({oid_name: obs_ids, aid_name: alt_ids})
   
        df = df.join(self.observations, how='left', on=oid_name)
        df = df.join(self.alternatives, how='left', on=aid_name)
        
        if (self.chosen_alternatives is not None):
            df['chosen'] = chosen
            df.sort_values([oid_name, 'chosen'], ascending=False, inplace=True)
        
        df.set_index([oid_name, aid_name], inplace=True)
        return df
        
    
    def _get_weights(self, obs_id):
        """
        Get sampling weights corresponding to a single observation id. 
        
        Parameters
        ----------
        observation_id : value from index of self.observations
        
        Expected class parameters
        -------------------------
        self.weights : pd.Series, callable, or None
        
        Returns
        -------
        pd.Series of weights
        
        """
        if (self.weights is None):
            return None
            
        elif (callable(self.weights)):
            # TO DO - implement
            pass
        
        elif (self.weights.shape[0] == self.alternatives.shape[0]):
            return self.weights/self.weights.sum()
        
        else:
            w = self.weights[self.weights.index.get_level_values('obs_id').isin([obs_id])]
            return w/w.sum()
            
    
    def _build_table_with_repeated_sample(self):
        """
        This handles the cases where we have to draw separate samples for each 
        observation -- sampling without replacement, or weights that apply to combinations
        of observation x alternative.
        
        Expected class parameters
        -------------------------
        self.observations : pd.DataFrame
        self.alternatives : pd.DataFrame
        self.chosen_alternatives : pd.Series or None
        self.sample_size : int
        self.replace : boolean
        self.weights : pd.Series, callable, or None
        self.random_state : NOT YET IMPLEMENTED

        """
        # TO DO - test this stuff
        
        obs_ids = np.repeat(self.observations.index.values, self.sample_size)
        alt_ids = []
        
        for obs_id in self.observations.index.values:
            sampled_alts = np.random.choice(self.alternatives.index.values,
                                            replace = self.replace,
                                            p = self._get_weights(obs_id),
                                            size = self.sample_size)
            alt_ids += sampled_alts

        # TO DO - implement chosen alternatives
        
        df = pd.DataFrame({'obs_id': obs_ids, 'alt_id': alt_ids})
        df.set_index(['obs_id', 'alt_id'], inplace=True)
   
        df = df.join(self.observations, how='left', on='obs_id')
        df = df.join(self.alternatives, how='left', on='alt_id')
        
        return df

    
    def _build_table(self):
        """
        Stub code to be refactored
        
        """
        # TO DO - finish implementing chosen alternatives
        # TO DO - implement availability (maybe put off until later)
        # TO DO - is availability the way to implement sampling buckets?
        # TO DO - implement interaction terms
        
        if (self.chosen_alternatives is not None):
            
            # TO DO - implement this differently for the no-sampling case?
        
            # Generate a binary chosen column: [1, 0, 0, 1, 0, 0, 1, 0, 0]
            _one_obs = np.append([1], np.repeat(0, repeats=self.sample_size-1))
            df['chosen'] = np.tile(_one_obs, reps=n_obs)
        
            # TO DO - place chosen alts (swap for no-sampling case)
                
        

    def to_frame(self):
        return self._merged_table


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
