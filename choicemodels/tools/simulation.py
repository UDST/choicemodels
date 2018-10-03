"""
Utilities for Monte Carlo simulation of choices.

"""
import numpy as np
import pandas as pd


def monte_carlo_choices(probabilities):
    """
    Monte Carlo simulation of choices for a set of K scenarios, each having different
    probability distributions (and potentially different alternatives). 
    
    Choices are independent and unconstrained, meaning that the same alternative can be 
    chosen in multiple scenarios.
    
    This function is equivalent to applying np.random.choice() to each of the K scenarios,
    but it's implemented as a single-pass matrix calculation. This is about 50x faster
    than using df.apply() or a loop. 
    
    If all the choice scenarios have the same probability distribution among alternatives,
    you don't need this function. You can use np.random.choice() with size=K, which will 
    be more efficient. (For example, that would work for a choice model whose expression 
    includes only attributes of the alternatives.)

    NOTE ABOUT THE INPUT FORMATS: It's important for the probabilities to be structured
    correctly. This is computationally expensive to verify, so you will not get a warning
    if it's wrong! (TO DO: we should provide an option to perform these checks, though)
    
    1. Probabilities (pd.Series) must include a two-level MultiIndex, the first level 
       representing the scenario (observation) id and the second the alternative id.

    2. Probabilities must be sorted so that each scenario's alternatives are consecutive.
    
    3. Each scenario must have the same number of alternatives. You can pad a scenario 
       with zero-probability alternatives if needed.
       
    4. Each scenario's alternative probabilities must sum to 1. 
    
    Parameters
    ----------
    probabilities: pd.Series
        List of probabilities for each observation (choice scenario) and alternative. 
        Please verify that the formatting matches the four requirements described above.
    
    Returns
    -------
    pd.Series
        List of chosen alternative id's, indexed with the observation id.
    
    """
    # TO DO - if input is a single-column dataframe, silently convert it to series

    obs_name, alts_name = probabilities.index.names

    obs = probabilities.index.get_level_values(0)
    alts = probabilities.index.get_level_values(1)
    
    num_obs = obs.unique().size
    num_alts = probabilities.size // num_obs
    
    # This Monte Carlo approach is adapted from urbansim.urbanchoice.mnl_simulate()
    probs = np.array(probabilities)
    cumprobs = probs.reshape((num_obs, num_alts)).cumsum(axis=1)
    
    # Simulate choice by subtracting a random float
    scaledprobs = np.subtract(cumprobs, np.random.rand(num_obs, 1))

    # Replace negative values with 0 and positive values with 1, then use argmax to
    # return the position of the first postive value
    choice_ix = np.argmax((scaledprobs + 1.0).astype('i4'), axis=1)
    choice_ix_1d = choice_ix + (np.arange(num_obs) * num_alts)
    
    choices = pd.DataFrame({obs_name: obs.values.take(choice_ix_1d),
                            alts_name: alts.values.take(choice_ix_1d)})
    
    return choices.set_index(obs_name)[alts_name]

