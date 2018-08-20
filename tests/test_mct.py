import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pytest

import choicemodels

d1 = {'oid': [0,1], 
      'obsval': [6,8],
      'choice': [1,2]}

d2 = {'aid': [0,1,2], 
      'altval': [10,20,30],
      'w': [1,1,100]}

obs = pd.DataFrame(d1).set_index('oid')
alts = pd.DataFrame(d2).set_index('aid')


# NO SAMPLING, TABLE FOR SIMULATION

mct = choicemodels.tools.MergedChoiceTable(obs, alts).to_frame()

df = pd.DataFrame({'oid': [0,0,0,1,1,1],
                   'aid': [0,1,2,0,1,2],
                   'obsval': [6,6,6,8,8,8],
                   'choice': [1,1,1,2,2,2],
                   'altval': [10,20,30,10,20,30],
                   'w': [1,1,100,1,1,100]}).set_index(['oid','aid'])

pd.testing.assert_frame_equal(mct, df)


# NO SAMPLING, TABLE FOR ESTIMATION

mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                             chosen_alternatives = 'choice').to_frame()

df = pd.DataFrame({'oid': [0,0,0,1,1,1],
                   'aid': [0,1,2,0,1,2],
                   'obsval': [6,6,6,8,8,8],
                   'altval': [10,20,30,10,20,30],
                   'w': [1,1,100,1,1,100],
                   'chosen': [0,1,0,0,0,1]}).set_index(['oid','aid'])

def test_one():
    pd.testing.assert_frame_equal(mct, df)


# REPLACEMENT, NO WEIGHTS, TABLE FOR SIMULATION

mct = choicemodels.tools.MergedChoiceTable(obs, alts, 
                             sample_size = 2).to_frame()

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

print(mct)







