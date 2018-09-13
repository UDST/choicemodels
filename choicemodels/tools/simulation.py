"""
Utilities for Monte Carlo simulation of choices.

"""
import numpy as np
import pandas as pd


def simulate_choices(probabilities):
    """
    Monte Carlo simulation of choices for a set of scenarios. 
    
    Alternatives and probabilities can vary across scenarios. Choices are independent and 
    unconstrained, meaning that one alternative can be chosen multiple times. (Support is 
    planned for capacity-constrained choices.)
    
    Does not support cases where the number of alternatives varies across choice 
    scenarios.
    
    Parameters
    ----------
    probabilities: pd.Series
        List of probabilities for each observation (choice scenario) and alternative. 
        Should contain a two-level MultiIndex, the first level representing the 
        observation id and the second the alternative id.
    
    Returns
    -------
    pd.Series
        List of chosen alternative id's, indexed with the observation id.
    
    """
    # TO DO 
    # - check input for consistent num_alts, probs that sum to 1
    # - if input is a single-column df, silently convert it to series
    # - MAKE SURE ALTERNATIVES ARE CONSECUTIVE
    
    obs_name, alts_name = probabilities.index.names

    obs = probabilities.index.get_level_values(0)
    alts = probabilities.index.get_level_values(1)
    
    num_obs = obs.unique().size
    num_alts = probabilities.size // num_obs
    
    # Generate choices by adapting an approach from UrbanSim MNL
    # https://github.com/UDST/choicemodels/blob/master/choicemodels/mnl.py#L578-L583
    probs = np.array(probabilities)
    cumprobs = probs.reshape((num_obs, num_alts)).cumsum(axis=1)
    rands = np.random.random(num_obs)
    diff = np.subtract(cumprobs.transpose(), rands).transpose()
    
    # The diff conversion replaces negative values with 0 and positive values with 1,
    # so that argmax can return the position of the first positive value
    choice_ix = np.argmax((diff + 1.0).astype('i4'), axis=1)
    choice_ix_1d = choice_ix + (np.arange(num_obs) * num_alts)
    
    choices = pd.DataFrame({obs_name: obs.values.take(choice_ix_1d),
                            alts_name: alts.values.take(choice_ix_1d)})
    
    return choices.set_index(obs_name)[alts_name]

