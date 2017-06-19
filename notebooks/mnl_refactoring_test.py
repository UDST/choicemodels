import numpy as np
import pandas as pd

from choicemodels import MultinomialLogit
from choicemodels.tools import MergedChoiceTable


tracts = pd.read_csv('../data/tracts.csv').set_index('full_tract_id')
trips = pd.read_csv('../data/trips.csv').set_index('place_id')

pd.set_option('display.float_format', lambda x: '%.3f' % x)

choosers = trips.loc[np.random.choice(trips.index, 500, replace=False)]
choosers = choosers.loc[choosers.trip_distance_miles.notnull()]

numalts = 10

merged = MergedChoiceTable(choosers = choosers, alternatives = tracts, 
						   chosen_alternatives = choosers.full_tract_id, 
						   sample_size = numalts)

'''
_, merged, chosen = interaction.mnl_interaction_dataset(
    choosers=choosers, alternatives=tracts, SAMPLE_SIZE=numalts, 
    chosenalts=choosers.full_tract_id)
'''

model_expression = "home_density + work_density + school_density"

model = MultinomialLogit(merged.to_frame(), merged.chosen, numalts, model_expression)

results = model.fit()

print(results)