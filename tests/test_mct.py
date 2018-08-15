import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
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

mct = choicemodels.tools.MCT(obs, alts).to_frame()

df = pd.DataFrame({'oid': [0,0,0,1,1,1],
                   'aid': [0,1,2,0,1,2],
                   'obsval': [6,6,6,8,8,8],
                   'choice': [1,1,1,2,2,2],
                   'altval': [10,20,30,10,20,30],
                   'w': [1,1,100,1,1,100]}).set_index(['oid','aid'])

pd.testing.assert_frame_equal(mct, df)


# NO SAMPLING, TABLE FOR ESTIMATION

mct = choicemodels.tools.MCT(obs, alts, 
                             chosen_alternatives = 'choice').to_frame()

df = pd.DataFrame({'oid': [0,0,0,1,1,1],
                   'aid': [0,1,2,0,1,2],
                   'obsval': [6,6,6,8,8,8],
                   'altval': [10,20,30,10,20,30],
                   'w': [1,1,100,1,1,100],
                   'chosen': [0,1,0,0,0,1]}).set_index(['oid','aid'])

pd.testing.assert_frame_equal(mct, df)


# RANDOM SAMPLING, REPLACEMENT, NO WEIGHTS, TABLE FOR SIMULATION

mct = choicemodels.tools.MCT(obs, alts, 
                             sample_size = 2).to_frame()

# TO DO - why is the 'choice' column missing?
assert len(mct) == 4
assert sum(mct.altval==30) < 4


# RANDOM SAMPLING, REPLACEMENT, NO WEIGHTS, TABLE FOR ESTIMATION

mct = choicemodels.tools.MCT(obs, alts, 
                             sample_size = 2,
                             chosen_alternatives = 'choice').to_frame()

assert len(mct) == 4
assert sum(mct.chosen==1) == 2


# RANDOM SAMPLING, REPLACEMENT, ALT-SPECIFIC WEIGHTS, TABLE FOR SIMULATION

mct = choicemodels.tools.MCT(obs, alts, 
                             sample_size = 2,
                             weights = 'w').to_frame()

assert len(mct) == 4
assert sum(mct.altval==30) > 2


# RANDOM SAMPLING, REPLACEMENT, ALT-SPECIFIC WEIGHTS, TABLE FOR ESTIMATION

mct = choicemodels.tools.MCT(obs, alts, 
                             sample_size = 2,
                             weights = 'w',
                             chosen_alternatives = 'choice').to_frame()



