"""
Utilities for constructing pairwise distance matrices and calculating which
geographies are within various distance bands of some reference geography
"""




import numpy as np
import pandas as pd
from itertools import tee
from scipy.spatial.distance import squareform, pdist




def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):
    """
    Vectorized function to calculate the great-circle distance between two
    points or between vectors of points. This function is borrowed from OSMnx:
    https://github.com/gboeing/osmnx

    Parameters
    ----------
    lat1 : float or vector of floats
    lng1 : float or vector of floats
    lat2 : float or vector of floats
    lng2 : float or vector of floats
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is
        meters)

    Returns
    -------
    distance : float or vector of floats
        distance or vector of distances from (lat1, lng1) to (lat2, lng2) in
        units of earth_radius
    """

    phi1 = np.deg2rad(90 - lat1)
    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(phi2))
    arc = np.arccos(cos)

    # return distance in units of earth_radius
    distance = arc * earth_radius
    return distance




def great_circle_distance_matrix(df, x, y, earth_radius=6371009,
                                 return_int=True):
    """
    Calculate a pairwise great-circle distance matrix from a DataFrame of
    points. Distances returned are in units of earth_radius (default is meters).

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract
        ID or parcel ID), represented by x and y coordinate columns
    x : str
        label of the x coordinate column in the DataFrame
    y : str
        label of the y coordinate column in the DataFrame
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is
        meters)
    return_int : bool
        if True, convert all distances to integers

    Returns
    -------
    pandas Series
        Multi-indexed distance vector in units of df's values, with top-level
        index representing "from" and second-level index representing "to".
    """

    # calculate pairwise great-circle distances between each row and every
    # other row
    df_dist_matrix = df.apply(lambda row: great_circle_vec(row[y], row[x], df[y], df[x], earth_radius=earth_radius),
                              axis='columns')

    # convert all distances to integers, if so configured
    if return_int:
        df_dist_matrix = df_dist_matrix.fillna(0).astype(int)

    # convert the distance matrix to a multi-indexed vector and return it
    labels = df.index.values
    df_dist_matrix.columns = labels
    df_dist_matrix.index = labels
    return df_dist_matrix.stack()




def euclidean_distance_matrix(df):
    """
    Calculate a pairwise euclidean distance matrix from a DataFrame of points.
    Distances returned are in units of x and y columns.

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract
        ID or parcel ID), represented by x and y coordinate columns

    Returns
    -------
    pandas Series
        Multi-indexed distance vector in units of df's values, with top-level
        index representing "from" and second-level index representing "to".
    """

    # calculate pairwise euclidean distances between all rows
    distance_vector = pdist(X=df, metric='euclidean')

    # convert the distance vector to a square distance matrix
    dist_matrix = squareform(distance_vector)

    # convert the distance matrix to a multi-indexed vector and return it
    labels = df.index.values
    df_dist_matrix = pd.DataFrame(data=dist_matrix, columns=labels, index=labels)
    return df_dist_matrix.stack()




def network_distance_matrix(df, x, y):
    """
    Calculate a pairwise network distance matrix from a DataFrame of points.

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract
        ID or parcel ID), represented by x and y coordinate columns
    x : str
        label of the x coordinate column in the DataFrame
    y : str
        label of the y coordinate column in the DataFrame

    Returns
    -------
    None
    """

    # placeholder until implementation of network distance matrix calculation
    raise NotImplementedError('network_distance_matrix is not currently implemented')




def distance_matrix(df, method='euclidean', x='lng', y='lat',
                    earth_radius=6371009, return_int=True):
    """
    Calculate a pairwise distance matrix from a DataFrame of two-dimensional
    points.

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract
        ID or parcel ID), represented by x and y coordinate columns
    method : str
        {'euclidean', 'greatcircle', 'network'}
        which algorithm to use for calculating pairwise distances
    x : str
        if method='greatcircle' or 'network', label of the x coordinate column
        in the DataFrame
    y : str
        if method='greatcircle' or 'network', label of the y coordinate column
        in the DataFrame
    earth_radius : numeric
        if method='greatcircle', radius of earth in units in which distance will
        be returned (default is meters)
    return_int : bool
        if method='greatcircle', if True, convert all distances to integers

    Returns
    -------
    pandas Series
        Multi-indexed distance vector in units of df's values, with top-level
        index representing "from" and second-level index representing "to".
    """

    if not df.index.is_unique:
        raise ValueError('The passed-in DataFrame must have a unique index')

    if method == 'euclidean':
        return euclidean_distance_matrix(df=df)
    elif method == 'greatcircle':
        return great_circle_distance_matrix(df=df, x=x, y=y,
                                            earth_radius=earth_radius,
                                            return_int=return_int)
    elif method == 'network':
        return network_distance_matrix(df=df, x=x, y=y)
    else:
        raise ValueError('argument `method` must be one of "euclidean", "greatcircle", or "network"')




def pairwise(iterable):
    """
    Iterate through a list, pairwise.

    Parameters
    ----------
    iterable : list-like
        the list-like object to iterate through pairwise

    Returns
    -------
    zip
        a zipped iterable of pairwise tuples
    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)




def distance_bands(dist_vector, distances):
    """
    Identify all geographies located within each distance band of each
    geography.

    The list of distances is treated pairwise to create distance bands, with
    the first element of each pair forming the band's inclusive lower limit and
    the second element of each pair forming the band's exclusive upper limit.
    For example, if distances=[0, 10, 30], band 0 will contain all geographies
    with a distance >= 0 and < 10 units (e.g., meters) from the reference
    geography, and band 1 will contain all geographies with a distance >= 10
    and < 30 units from the reference geography.

    To make the final distance band include all geographies beyond a certain
    distance, make the final value in the distances list np.inf.

    Parameters
    ----------
    dist_vector : pandas Series
        Multi-indexed distance vector in units of df's values, with top-level
        index representing "from" and second-level index representing "to".
    distances : list
        a list of distance band increments

    Returns
    -------
    pandas Series
        a series multi-indexed by geography ID and distance band number, with
        values of arrays of geography IDs with the corresponding distances from
        that ID
    """

    # loop through each row in distance matrix, identifying all geographies
    # within each distance band of the row's geography
    bands = {}
    dist_matrix = dist_vector.unstack()
    for _, row in dist_matrix.iterrows():
        bands[row.name] = {}

        # for each distance band
        for band_number, (dist1, dist2) in enumerate(pairwise(distances)):

            # find all the geographies within this distance band
            mask = (row >= dist1) & (row < dist2)
            place_ids = row[mask].index.values

            # store value as array of geography IDs keyed by reference geography
            # ID and distance band number
            bands[row.name][band_number] = place_ids

    # convert geography bands to a series multi-indexed by geography ID and
    # distance band number
    return pd.DataFrame(bands).T.stack()
