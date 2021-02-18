import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pytest

import choicemodels
from choicemodels.tools import MergedChoiceTable


@pytest.fixture
def obs():
    d1 = {'oid': [0,1], 
          'obsval': [6,8],
          'choice': [1,2]}
    
    return pd.DataFrame(d1).set_index('oid')

@pytest.fixture
def alts():
    d2 = {'aid': [0,1,2], 
          'altval': [10,20,30],
          'w': [1,1,100]}

    return pd.DataFrame(d2).set_index('aid')


@pytest.fixture
def stratified_alts():
    d2 = {'aid': [0,1,2,3,4,5,6,7,8,9], 
          'altval': [10,20,30,40,50,60,70,80,90,100],
          'stratum': [1,2,3,4,5,1,2,3,4,5]}

    return pd.DataFrame(d2).set_index('aid')


def test_mergedchoicetable(obs, alts, stratified_alts):
    # NO SAMPLING, TABLE FOR SIMULATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts).to_frame()

    assert list(mct.columns.sort_values()) ==  list(sorted(['obsval', 'altval', 
                                                            'w', 'choice']))


    # NO SAMPLING, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 chosen_alternatives = 'choice').to_frame()

    assert list(mct.columns.sort_values()) == list(sorted(['obsval', 'altval', 
                                                           'w', 'chosen']))
    
    # STRATIFIED SAMPLING
    mct = choicemodels.tools.MergedChoiceTable(
        obs, stratified_alts, sample_size=5, sampling_regime='stratified',
        strata='stratum').to_frame()

    for obs_id, obs_df in mct.groupby(level=0):
        assert len(obs_df['stratum'].unique()) == 5

    # REPLACEMENT, NO WEIGHTS, TABLE FOR SIMULATION

    mct = choicemodels.tools.MergedChoiceTable(
        obs, alts, sample_size=2).to_frame()

    assert len(mct) == 4
    assert sum(mct.altval==30) < 4


    # REPLACEMENT, NO WEIGHTS, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 chosen_alternatives = 'choice').to_frame()

    assert len(mct) == 4
    assert sum(mct.chosen==1) == 2


    # REPLACEMENT, ALT-SPECIFIC WEIGHTS, TABLE FOR SIMULATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 weights = 'w').to_frame()

    assert len(mct) == 4
    assert sum(mct.altval==30) > 2


    # REPLACEMENT, ALT-SPECIFIC WEIGHTS, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 weights = 'w',
                                 chosen_alternatives = 'choice').to_frame()


    # NO REPLACEMENT, NO WEIGHTS, TABLE FOR SIMULATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 3,
                                 replace = False).to_frame()

    assert len(mct) == 6
    assert len(mct.loc[0].index.unique()) == 3


    # NO REPLACEMENT, NO WEIGHTS, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 3,
                                 replace = False,
                                 chosen_alternatives = 'choice').to_frame()

    assert len(mct) == 6
    assert len(mct.loc[0].index.unique()) == 3
    assert sum(mct.chosen==1) == 2


    # NO REPLACEMENT, ALT-SPECIFIC WEIGHTS, TABLE FOR SIMULATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 replace = False,
                                 weights = 'w').to_frame()

    assert len(mct) == 4
    assert len(mct.loc[0].index.unique()) == 2
    assert sum(mct.altval==30) == 2


    # NO REPLACEMENT, ALT-SPECIFIC WEIGHTS, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 replace = False,
                                 weights = 'w',
                                 chosen_alternatives = 'choice').to_frame()

    assert len(mct) == 4
    assert len(mct.loc[0].index.unique()) == 2
    assert sum(mct.altval==30) == 2
    assert sum(mct.chosen==1) == 2


    # REPLACEMENT, OBS-ALT INTERACTION WEIGHTS, TABLE FOR SIMULATION

    w = {'w': [1,1,100,25,25,25],
         'oid': [0,0,0,1,1,1],
         'aid': [0,1,2,0,1,2]}

    wgt = pd.DataFrame(w).set_index(['oid','aid']).w

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 replace = True,
                                 weights = wgt).to_frame()


    # REPLACEMENT, OBS-ALT INTERACTION WEIGHTS, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 replace = True,
                                 weights = wgt,
                                 chosen_alternatives = 'choice').to_frame()


    # NO REPLACEMENT, OBS-ALT INTERACTION WEIGHTS, TABLE FOR SIMULATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 replace = False,
                                 weights = wgt).to_frame()


    # NO REPLACEMENT, OBS-ALT INTERACTION WEIGHTS, TABLE FOR ESTIMATION

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 replace = False,
                                 weights = wgt,
                                 chosen_alternatives = 'choice').to_frame()


def test_no_alternatives(obs, alts):
    """
    Empty alternatives should produce empty choice table.
    
    """
    mct = MergedChoiceTable(obs, pd.DataFrame()).to_frame()
    assert len(mct) == 0


def test_no_choosers(obs, alts):
    """
    Empty observations should produce empty choice table.
    
    """
    mct = MergedChoiceTable(pd.DataFrame(), alts).to_frame()
    assert len(mct) == 0


def test_dupe_column(obs, alts):
    """
    Duplicate column names should raise an error.
    
    """
    obs['save_the_whales'] = None
    alts['save_the_whales'] = None
    
    try:
        MergedChoiceTable(obs, alts)
    except ValueError as e:
        print(e)


def test_multiple_dupe_columns(obs, alts):
    """
    Duplicate column names should raise an error. This covers the case of multiple
    columns, and the case of an index conflicting with a non-index.
    
    """
    obs['save_the_whales'] = None
    alts['save_the_whales'] = None
    alts[obs.index.name] = None
    
    try:
        MergedChoiceTable(obs, alts)
    except ValueError as e:
        print(e)


def test_join_key_name_conflict(obs, alts):
    """
    Duplicate column names are not allowed, except for the join key -- it's fine for the 
    chosen_alternatives column in the observations to have the same name as the index of
    the alternatives. This test should run without raising an error.
    
    """
    obs[alts.index.name] = obs.choice
    MergedChoiceTable(obs, alts, chosen_alternatives=alts.index.name)


def test_obs_id_property(obs, alts):
    """
    Observation id should be available for a merged table.
    
    """
    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 chosen_alternatives = 'choice')
    
    assert(mct.observation_id_col == 'oid')


def test_alt_id_property(obs, alts):
    """
    Alternative id should be available for a merged table.
    
    """
    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 chosen_alternatives = 'choice')
    
    assert(mct.alternative_id_col == 'aid')


def test_choice_col_property(obs, alts):
    """
    Choice column property should be present if applicable, or None.
    
    """
    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 chosen_alternatives = 'choice')
    assert(mct.choice_col == 'chosen')

    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2)
    assert(mct.choice_col == None)


def test_from_df(obs, alts):
    """
    MCT creation from a dataframe should work smoothly.
    
    """
    df = choicemodels.tools.MergedChoiceTable(obs, alts, 
                                 sample_size = 2,
                                 chosen_alternatives = 'choice').to_frame()
    
    mct = choicemodels.tools.MergedChoiceTable.from_df(df)
    
    assert(df.equals(mct.to_frame()))

