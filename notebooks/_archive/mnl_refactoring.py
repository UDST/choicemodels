import numpy as np
import pandas as pd

from choicemodels import MultinomialLogit
from choicemodels.tools import MergedChoiceTable
from collections import OrderedDict


tracts = pd.read_csv('../data/tracts.csv').set_index('full_tract_id')
trips = pd.read_csv('../data/trips.csv').set_index('place_id')

pd.set_option('display.float_format', lambda x: '%.3f' % x)

choosers = trips.loc[np.random.choice(trips.index, 500, replace=False)]
choosers = choosers.loc[choosers.trip_distance_miles.notnull()]

numalts = 10

merged = MergedChoiceTable(observations = choosers,
						   alternatives = tracts,
                           chosen_alternatives = choosers.full_tract_id,
                           sample_size = numalts)

model_expression = "home_density + work_density + school_density"

model = MultinomialLogit(merged.to_frame(),
						 merged.observation_id_col,
						 merged.choice_col,
						 model_expression)

results = model.fit()

results.report_fit()

"""
model_expression = OrderedDict([('home_density', 'all_same'),
								('work_density', 'all_same'),
								('school_density', 'all_same')])

model = MultinomialLogit(data = merged.to_frame(),
						 observation_id_col = merged.observation_id_col,
						 alternative_id_col = merged.alternative_id_col,
						 choice_col = merged.choice_col,
						 model_expression = model_expression)

results = model.fit()
print(results.print_summaries())
"""
