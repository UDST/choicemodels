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
    but it's implemented as a single-pass matrix calculation. When the number of scenarios 
    is large, this is about 50x faster than using df.apply() or a loop.
    
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


def iterative_lottery_choices(choosers, alternatives, mct_callable, probs_callable, 
        alt_capacity=None, chooser_size=None, max_iter=None):
    """
    Monte Carlo simulation of choices for a set of choice scenarios where (a) the 
    alternatives have limited capacity and (b) the choosers have varying probability 
    distributions over the alternatives. 
    
    Effectively, we simulate the choices sequentially, each time removing the chosen
    alternative or reducing its available capacity. (It's actually done in batches for
    better performance, but the outcome is equivalent.) This requires sampling 
    alternatives and calculating choice probabilities multiple times, which is why
    callables for those actions are required inputs.
    
    Chooser priority is randomized. Capacities can be specified as counts (number of 
    choosers that can be accommodated) or as amounts (e.g. square footage) with 
    corresponding chooser sizes. If total capacity is insufficient to accommodate all the 
    choosers, as many choices will be simulated as possible.
    
    Note that if all the choosers are the same size and have the same probability 
    distribution over alternatives, you don't need this function.
    
    Parameters
    ----------
    choosers : pd.DataFrame
        Table with one row for each chooser or choice scenario, with unique ID's in the
        index field. Additional columns can contain fixed attributes of the choosers.
        (Reserved column names: '_size'.)
    
    alternatives : pd.DataFrame
        Table with one row for each alternative, with unique ID's in the index field.
        Additional columns can contain fixed attributes of the alternatives. (Reserved 
        column names: '_capacity'.)
    
    mct_callable : callable
        Callable that samples alternatives to generate a table of choice scenarios. It 
        should accept subsets of the choosers and alternatives tables and return a 
        choicemodels.tools.MergedChoiceTable.
    
    probs_callable : callable
        Callable that generates predicted probabilities for a table of choice scenarios.
        It should accept a choicemodels.tools.MergedChoiceTable and return a pd.Series
        with indexes matching the input.
    
    alt_capacity : str, optional
        Name of a column in the alternatives table that expresses the capacity of 
        alternatives. If not provided, each alternative is interpreted as accommodating a
        single chooser.
    
    chooser_size : str, optional
        Name of a column in the choosers table that expresses the size of choosers. 
        Choosers might have varying sizes if the alternative capacities are amounts 
        rather than counts -- e.g. square footage or employment capacity. Chooser sizes 
        must be in the same units as alternative capacities. If not provided, each chooser
        has a size of 1. 
    
    max_iter : int or None, optional
        Maximum number of iterations. If None (default), the algorithm will iterate until 
        all choosers are matched or no alternatives remain.

    Returns
    -------
    pd.Series
        List of chosen alternative id's, indexed with the chooser (observation) id. 
            
    """
    # TO DO - how does MCT handle sample size greater than the number of available alts?
    
    if alt_capacity is None:
        alt_capacity = '_capacity'
        alternatives.loc[:,alt_capacity] = 1
    
    if chooser_size is None:
        chooser_size = '_size'
        choosers.loc[:,chooser_size] = 1
    
    alts = alternatives
    capacity, size = (alt_capacity, chooser_size)
    
    len_choosers = len(choosers)
    valid_choices = pd.Series()
    iter = 0
    
    while (len(valid_choices) < len_choosers):
        iter += 1
        if max_iter is not None:
            if (iter > max_iter):
                break

        mct = mct_callable(choosers.sample(frac=1), alts)
        
        if len(mct.to_frame()) == 0:
            print("No valid alternatives for the remaining choosers")
            break
        
        probs = probs_callable(mct)
        choices = pd.DataFrame(monte_carlo_choices(probs))
    
        # join capacities and sizes
        oid, aid = (mct.observation_id_col, mct.alternative_id_col)
        c = choices.join(alts[capacity], on=aid).join(choosers[size], on=oid)
        c.loc[:,'_cumsize'] = c.groupby(aid)[size].cumsum()
    
        # save valid choices
        c_valid = (c._cumsize <= c[capacity])
        valid_choices = pd.concat([valid_choices, c[aid].loc[c_valid]])
        
        print("Iteration {}: {} of {} valid choices".format(iter, len(valid_choices), 
                len_choosers))
        
        # update choosers and alternatives
        choosers = choosers.drop(c.loc[c_valid].index.values)
        # print("{} remaining choosers".format(len(choosers)))
        
        placed_capacity = c.loc[c_valid].groupby(aid)._cumsize.max()
        alts.loc[:,capacity] = alts[capacity].subtract(placed_capacity, fill_value=0)

        full = alts.loc[alts[capacity] == 0]
        alts = alts.drop(full.index)
        # print("{} remaining alternatives".format(len(alts)))
        
    return valid_choices
    
    
    
    
    