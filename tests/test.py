def test_import():
    import choicemodels

def test_simple_estimation():
    import choicemodels
    import numpy as np
    import pandas as pd
    from collections import OrderedDict
    endog = np.random.randint(2, size=50)
    exog = np.random.rand(50, 5)
    m = choicemodels.Logit(endog, exog)
    results = m.fit()
    results.summary()
