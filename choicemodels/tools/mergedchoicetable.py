"""
Utilities for generating merged tables of choosers and alternatives, including extensive
sampling functionality.

"""
import numpy as np
import pandas as pd


class MergedChoiceTable(object):
    """
    Generates a merged long-format table of observations (choosers) and alternatives, for 
    discrete choice model estimation or simulation. 
    
    Supports random sampling of alternatives (uniform or weighted). Supports sampling with
    or without replacement. Supports merging observations and alternatives without 
    sampling them. Supports alternative-specific weights, as well as interaction weights
    that depend on both the observation and alternative. Supports automatic merging of 
    interaction terms onto the final data table.
    
    Support is PLANNED for specifying availability of alternatives, specifying random 
    state, and passing interaction-type parameters as callable generator functions.
    
    Does NOT support cases where the number of alternatives in the final table varies 
    across observations.

    Reserved column names: 'chosen'.
    
    Parameters
    ----------
    observations : pandas.DataFrame
        Table with one row for each chooser or choice scenario, with unique ID's in the 
        index field. Additional columns can contain fixed attributes of the choosers. 
        Index name is set to 'obs_id' if none provided. All observation/alternative 
        column names must be unique except for the join key.

    alternatives : pandas.DataFrame
        Table with one row for each alternative, with unique ID's in the index field.
        Additional columns can contain fixed attributes of the alternatives. Index name
        is set to 'alt_id' if none provided. All observation/alternative column names
        must be unique except for the join key.

    chosen_alternatives : str or pandas.Series, optional
        List of the alternative ID selected in each choice scenario. (This is required for
        preparing estimation data, but not for simulation data.) If str, interpreted as a
        column name from the observations table. If Series, it will be joined onto the
        obserations table before processing. The column will be dropped from the merged 
        table and replaced with a binary column named 'chosen'.

    sample_size : int, optional
        Number of alternatives to sample for each choice scenario. If 'None', all of the 
        alternatives will be available for each chooser in the merged table. The sample 
        size includes the chosen alternative, if applicable. If replace=False, the sample
        size must be less than or equal to the total number of alternatives. 

    replace : boolean, optional
        Whether to sample alternatives with or without replacement, at the level of a 
        single chooser or choice scenario. If replace=True (default), alternatives may
        appear multiple times in a single choice set. If replace=False, an alternative
        will appear at most once per choice set. Sampling with replacement is much more
        efficient, so setting replace=False may have performance implications if there are
        very large numbers of observations or alternatives.        
    
    weights : str, pandas.Series, optional
        Numerical weights to apply when sampling alternatives. If str, interpreted as a 
        column from the alternatives table. If Series, it can contain either (a) one 
        weight for each alternative or (b) one weight for each combination of observation 
        and alternative. The former should include a single index with ID's from the 
        alternatives table. The latter should include a MultiIndex with the first level 
        corresponding to the observations table and the second level corresponding to the 
        alternatives table. If callable, it should accept two arguments (obs_id, alt_id) 
        and return the corresponding weight.
        
        TO DO - accept weights specified with respect to derivative characteristics, like
        how the interaction terms work (for example weights could be based on home census
        tract rather than observation id if there are multiple observations per tract)
        
        TO DO - implement support for a callable
    
    availability : pandas.Series or callable, optional (NOT YET IMPLEMENTED)
        Binary representation of the availability of alternatives. Specified and applied 
        similarly to the weights.
    
    interaction_terms : pandas.Series, pandas.DataFrame, or list of either, optional
        Additional column(s) of interaction terms whose values depend on the combination 
        of observation and alternative, to be merged onto the final data table. If passed
        as a Series or DataFrame, it should include a two-level MultiIndex. One level's 
        name and values should match an index or column from the observations table, and 
        the other should match an index or column from the alternatives table. 
        
        TO DO - implement support for a callable
            
    random_state : NOT YET IMPLEMENTED
        Representation of random state, for replicability of the sampling.

    """
    def __init__(self, observations, alternatives, chosen_alternatives=None,
                 sample_size=None, replace=True, weights=None, availability=None,
                 interaction_terms=None, random_state=None):
        
        # Standardize and validate the inputs...
        
        if isinstance(sample_size, float):
            sample_size = int(sample_size)
        
        if (sample_size is not None):
            if (sample_size <= 0):
                raise ValueError("Cannot sample {} alternatives; to run without sampling "
                        "leave sample_size=None".format(sample_size))

            # TO DO - should probably just return as many alternatives as we can (and wait 
            # to evaluate this until after evaluating the sampling filters)
            if (replace == False) & (sample_size > alternatives.shape[0]):
                raise ValueError("Cannot sample without replacement with sample_size {} "
                        "and n_alts {}".format(sample_size, alternatives.shape[0]))
        
        # TO DO - check that dfs have unique indexes
        # TO DO - check that chosen_alternatives correspond correctly to other dfs
        # TO DO - same with weights (could join onto other tables and then split off)
        
        # Normalize chosen_alternatives to a pd.Series
        if (chosen_alternatives is not None) & isinstance(chosen_alternatives, str):
            chosen_alternatives = observations[chosen_alternatives].copy()
            observations = observations.drop(chosen_alternatives.name, axis='columns')
            chosen_alternatives.name = '_' + alternatives.index.name  # avoids conflicts
        
        # Allow missing obs and alts, to support .from_df() constructor     
        if (observations is not None):
        
            # Provide default names for observation and alternatives id's
        
            if (observations.index.name == None):
                observations.index.name = 'obs_id'
        
            if (alternatives.index.name == None):
                alternatives.index.name = 'alt_id'
        
            # Check for duplicate column names
            obs_cols = list(observations.columns) + list(observations.index.names)
            alt_cols = list(alternatives.columns) + list(alternatives.index.names)
            dupes = set(obs_cols) & set(alt_cols)
        
            if len(dupes) > 0:
                raise ValueError("Both input tables contain column {}. Please ensure "
                                 "column names are unique before merging".format(dupes))
        
        # Normalize weights to a pd.Series
        if (weights is not None) & isinstance(weights, str):
            weights = alternatives[weights]
        
        weights_1d = False
        weights_2d = False
        if (weights is not None):
            # TO DO - would be nicer to test using the dimensionality of the index and 
            #   then automatically filter for applicable weights if there are too many
            if (len(weights) == len(alternatives)):
                weights_1d = True
            elif (len(weights) == len(observations) * len(alternatives)):
                weights_2d = True
            else:
                raise ValueError("Length of weights is not aligned with length of "
                                 "alternatives and/or observations")
        
        # TO DO - if user passes a single-column df of weights instead of a series, we
        #   should just silently convert it
        
        self.observations = observations
        self.alternatives = alternatives
        self.chosen_alternatives = chosen_alternatives
        self.sample_size = sample_size
        self.replace = replace
        self.weights = weights
        self.interaction_terms = interaction_terms
        self.random_state = random_state

        self.weights_1d = weights_1d
        self.weights_2d = weights_2d
        
        # Build choice table...
        # Allow missing obs and alts, to support .from_df() constructor     
        if (observations is not None):

            if (len(observations) == 0) or (len(alternatives) == 0):
                self._merged_table = pd.DataFrame()
        
            elif (sample_size is None):
                self._merged_table = self._build_table_without_sampling()
        
            else:
                self._merged_table = self._build_table()
        
        
    @classmethod
    def from_df(cls, df):
        """
        Create a MergedChoiceTable instance from a pre-generated DataFrame.

        Each chooser's rows should be contiguous. If applicable, the chosen alternative
        should be listed first. This ordering is used by MergedChoiceTable.to_frame(),
        and appears to be an undocumented requirement of the legacy MNL code.

        Parameters
        ----------
        df : pandas.DataFrame
            Table with a two-level MultiIndex where the first level corresponds to the
            index of the observations and the second to the index of the alternatives.
            May include a binary column named 'chosen' indicating observed choices.
        
        Returns
        -------
        MergedChoiceTable

        """
        obj = cls(observations = None, alternatives = None)
        obj._merged_table = df
        
        # TO DO: sort the dataframe so that rows are automatically in a consistent order
        
        return obj


    def _merge_interaction_terms(self, df):
        """
        Merges interaction terms (if they exist) onto the input DataFrame. 
        
        Parameters
        ----------
        df : pd.DataFrame
            Should contain two columns whose names match the index levels of the 
            interaction data.
        
        Expected class parameters
        -------------------------
        self.interaction_terms : pd.Series or pd.DataFrame, optional
            Should have a two-level, named MultiIndex. If self.interaction_terms is None, 
            function returns the input DataFrame.
            
        Returns
        -------
        pd.DataFrame
            Same format as input, with interaction term column(s) added.
        
        """
        if (self.interaction_terms is None):
            return df

        if not isinstance(self.interaction_terms, list):
            self.interaction_terms = [self.interaction_terms]

        for intx_table in self.interaction_terms:
            df = df.join(pd.DataFrame(intx_table), how='left', 
                         on=intx_table.index.names)
        
        return df
    
    
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
        
        df = self._merge_interaction_terms(df)
        df.set_index([oid_name, aid_name], inplace=True)
        return df

    
    def _get_availability(self, obs_id):
        """
        Get alternative availability for a single observation id. For now, this just 
        checks whether the chosen alternative is known and if so makes it unavailable.
        
        Parameters
        ----------
        observation_id : value from index of self.observations
        
        Expected class parameters
        -------------------------
        self.alternatives : pd.DataFrame
        self.chosen_alternatives : pd.Series or None
        
        Returns
        -------
        list of booleans
        
        """
        # TO DO - seems inefficient?

        a = np.repeat(True, self.alternatives.shape[0])
        
        if (self.chosen_alternatives is not None):
            a = (self.alternatives.index != self.chosen_alternatives[obs_id])

        return a        
    
    
    def _get_weights(self, obs_id):
        """
        Get sampling weights corresponding to a single observation id.
        
        Parameters
        ----------
        observation_id : value from index of self.observations
        
        Expected class parameters
        -------------------------
        self.weights : pd.Series, callable, or None
        self.chosen_alternatives : pd.Series or None
        self.weights_1d : boolean
        self.weights_2d : boolean
        
        Returns
        -------
        pd.Series of weights
        
        """
        if (self.weights is None):
            return
            
        elif (callable(self.weights)):
            # TO DO - implement
            pass
        
        elif (self.weights_1d == True):
            return self.weights
        
        elif (self.weights_2d == True):
            return self.weights.loc[obs_id]
        
        else:
            raise ValueError  # unexpected inputs
                    

    def _build_table(self):
        """
        Build and return the merged choice table. This handles all cases where sampling
        is performed.
        
        Expected class parameters
        -------------------------
        self.observations : pd.DataFrame
        self.alternatives : pd.DataFrame
        self.chosen_alternatives : pd.Series or None
        self.sample_size : int
        self.replace : boolean
        self.weights : pd.Series, callable, or None
        self.random_state : NOT YET IMPLEMENTED
        
        Returns
        -------
        pd.DataFrame

        """
        n_obs = self.observations.shape[0]
        
        oid_name = self.observations.index.name
        aid_name = self.alternatives.index.name
        
        samp_size = self.sample_size
        if (self.chosen_alternatives is not None):
            samp_size = self.sample_size - 1
        
        obs_ids = np.repeat(self.observations.index.values, samp_size)
        
        # SINGLE SAMPLE: this covers cases where we can draw a single, large sample of 
        # alternatives and distribute them among the choosers, e.g. sampling without 
        # replacement, with optional alternative-specific weights but NOT weights that 
        # apply to combinations of observation x alternative
        
        if (self.replace == True) & (self.weights is None):
            
            alt_ids = np.random.choice(self.alternatives.index.values, 
                                       replace = True,
                                       size = n_obs * samp_size)
        
        elif (self.replace == True) & (self.weights_1d == True):
            
            alt_ids = np.random.choice(self.alternatives.index.values, 
                                       replace = True,
                                       p = self.weights/self.weights.sum(),
                                       size = n_obs * samp_size)
        
        # REPEATED SAMPLES: this covers cases where we have to draw separate samples for
        # each observation, e.g. sampling without replacement, or weights that apply to
        # combinations of observation x alternative
        
        elif (self.replace == False) | (self.weights_2d == True):
            
            alt_ids = []

            for obs_id in self.observations.index.values:
                a = self._get_availability(obs_id)                
                available_alts = self.alternatives.loc[a].index.values
                
                w = self._get_weights(obs_id)
                if (w is not None):
                    w = w.loc[a]/w.loc[a].sum()                
                
                sampled_alts = np.random.choice(available_alts, replace=self.replace, 
                                                p=w, size=samp_size).tolist()
                alt_ids += sampled_alts

        
        # Append chosen ids if applicable
        if (self.chosen_alternatives is not None):
            obs_ids = np.append(obs_ids, self.observations.index.values)
            alt_ids = np.append(alt_ids, self.chosen_alternatives)
            chosen = np.append(np.repeat(0, samp_size * n_obs), np.repeat(1, n_obs))
        
        df = pd.DataFrame({oid_name: obs_ids, aid_name: alt_ids})
   
        df = df.join(self.observations, how='left', on=oid_name)
        df = df.join(self.alternatives, how='left', on=aid_name)
        
        if (self.chosen_alternatives is not None):
            df['chosen'] = chosen
            df.sort_values([oid_name, 'chosen'], ascending=False, inplace=True)
        
        df = self._merge_interaction_terms(df)
        df.set_index([oid_name, aid_name], inplace=True)
        return df
        
    
    def to_frame(self):
        """
        Long-format DataFrame of the merged table. The rows representing alternatives for
        a particular chooser are contiguous, with the chosen alternative listed first if
        applicable. (Unless no sampling is performed, in which case the alternatives are
        listed in order.) The DataFrame includes a two-level MultiIndex. The first level
        corresponds to the index of the observations table and the second to the index of 
        the alternatives table. 
        
        Returns
        -------
        pandas.DataFrame

        """
        return self._merged_table

    
    @property
    def observation_id_col(self):
        """
        Name of column in the merged table containing the observation id. Name and values 
        will match the index of the observations table.
        
        Returns
        -------
        str

        """
        return self._merged_table.index.names[0]

    
    @property
    def alternative_id_col(self):
        """
        Name of column in the merged table containing the alternative id. Name and values
        will match the index of the alternatives table.

        Returns
        -------
        str

        """
        return self._merged_table.index.names[1]

    
    @property
    def choice_col(self):
        """
        Name of the generated column containing a binary representation of whether each
        alternative was chosen in the given choice scenario, if applicable.

        Returns
        -------
        str or None

        """
        if ('chosen' in self._merged_table.columns):
            return 'chosen'
        
        else:
            return None

