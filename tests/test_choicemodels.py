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

def test_distance_matrix():
    import pandas as pd, numpy as np
    from choicemodels.tools import distancematrix as dm
    df = pd.DataFrame()
    df['lat'] = [37.86, 37.85, 37.84, 37.87, 37.88]
    df['lng'] = [-122.27, -122.28, -122.26, -122.29, -122.25]
    dists_eu = dm.distance_matrix(df, method='euclidean')
    dists_gc = dm.distance_matrix(df, method='greatcircle')
    distances = [0, 2000, 4000, np.inf]
    db = dm.distance_bands(dists_gc, distances)
