import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import choicemodels

obs = pd.DataFrame({'oid': [0,1], 'obsval': [6,8]}).set_index('oid')
alts = pd.DataFrame({'aid': [0,1,2], 'altval': [10,20,30], 
                     'w': [1,1,100]}).set_index('aid')

# No sampling

mct = choicemodels.tools.MCT(obs, alts, sample_size=None).to_frame()

df = pd.DataFrame({'oid': [0,0,0,1,1,1],
                   'aid': [0,1,2,0,1,2],
                   'obsval': [6,6,6,8,8,8],
                   'altval': [10,20,30,10,20,30],
                   'w': [1,1,100,1,1,100]}).set_index(['oid','aid'])

pd.testing.assert_frame_equal(mct, df)