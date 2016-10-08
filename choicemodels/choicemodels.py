# ChoiceModels
# See full license in LICENSE

import numpy as np
import pandas as pd
import pylogit
import statsmodels.api as sm


def convert_wide_to_long(*args, **kwargs):
    return pylogit.convert_wide_to_long(*args, **kwargs)


class Logit(object):
    """
    Wraps the functionality of statsmodels.discrete.discrete_model.Logit()
    
    """
    def __init__(self, *args, **kwargs):
        self.wrapped_model = sm.Logit(*args, **kwargs)
        return

    def fit(self, *args, **kwargs):
        return self.wrapped_model.fit(*args, **kwargs)
        
    def from_formula(self, *args, **kwargs):
        return self.wrapped_model.from_formula(*args, **kwargs)
        
    def predict(self, *args, **kwargs):
        return self.wrapped_model.predict(*args, **kwargs)
        
    def extra(self):
        print("testing extra methods")
        return
        

class MNLogit(object):
    """
    Wraps the functionality of pylogit.conditional_logit.MNL()
    
    """
    def __init__(self, *args, **kwargs):
        self.wrapped_model = pylogit.create_choice_model(model_type="MNL", *args, **kwargs)
        return
        
    def fit_mle(self, *args, **kwargs):
        self.wrapped_model.fit_mle(*args, **kwargs)
        return CMResults(self.wrapped_model)


class CMResults(object):
    """
    Stores estimation results
    
    """
    def __init__(self, model):
        self.wrapped_model = model
        return
        
    def summary(self):
        return self.wrapped_model.get_statsmodels_summary()




