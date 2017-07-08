"""
Utilities for constructing pairwise distance matrices
"""




import numpy as np
import pandas as pd
from itertools import tee
from scipy.spatial.distance import squareform, pdist




def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):
    """
    Vectorized function to calculate the great-circle distance between two points or between vectors
    of points. This function is borrowed from OSMnx.

    Parameters
    ----------
    lat1 : float or array of float
    lng1 : float or array of float
    lat2 : float or array of float
    lng2 : float or array of float
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is meters)

    Returns
    -------
    distance : float or array of float
        distance or vector of distances from (lat1, lng1) to (lat2, lng2) in units of earth_radius
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




def great_circle_distance_matrix(df, x, y, earth_radius=6371009, return_int=True):
    """
    Calculate a pairwise great-circle distance matrix from a DataFrame of points.
    Distances returned are in units of earth_radius (default is meters).

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract ID
        or parcel ID), represented by x and y coordinate columns
    place_id : str
        label of the place_id column in the DataFrame
    x : str
        label of the x coordinate column in the DataFrame
    y : str
        label of the y coordinate column in the DataFrame
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is meters)
    return_int : bool
        if True, convert all distances to integers

    Returns
    -------
    df_dist_matrix : pandas DataFrame
        square distance matrix in units of earth_radius
    """

    # calculate pairwise great-circle distances between each row and every other row
    df_dist_matrix = df.apply(lambda row: great_circle_vec(row[y], row[x], df[y], df[x], earth_radius=earth_radius),
                              axis='columns')

    # convert all distances to integers, if so configured
    if return_int:
        df_dist_matrix = df_dist_matrix.fillna(0).astype(int)

    # set matrix's column and row indices to the dataframe's row labels, then return
    labels = df.index.values
    df_dist_matrix.columns = labels
    df_dist_matrix.index = labels
    return df_dist_matrix




def euclidean_distance_matrix(df, x, y):
    """
    Calculate a pairwise euclidean distance matrix from a DataFrame of points.
    Distances returned are in units of x and y columns.

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract ID
        or parcel ID), represented by x and y coordinate columns
    x : str
        label of the x coordinate column in the DataFrame
    y : str
        label of the y coordinate column in the DataFrame

    Returns
    -------
    df_dist_matrix : pandas DataFrame
        square distance matrix in units of x and y
    """

    # calculate pairwise euclidean distances between all rows
    distance_vector = pdist(X=df, metric='euclidean')

    # convert the distance vector to a square distance matrix
    dist_matrix = squareform(distance_vector)

    # convert the distance matrix to a dataframe and return it
    labels = df.index.values
    df_dist_matrix = pd.DataFrame(data=dist_matrix, columns=labels, index=labels)
    return df_dist_matrix




def network_distance_matrix(df, x, y):
    """
    Calculate a pairwise network distance matrix from a DataFrame of points.

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract ID
        or parcel ID), represented by x and y coordinate columns
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




def distance_matrix(df, method='euclidean', x='lng', y='lat', earth_radius=6371009, return_int=True):
    """
    Calculate a pairwise distance matrix from a DataFrame of two-dimensional points.

    Parameters
    ----------
    df : pandas DataFrame
        a DataFrame of points, uniquely indexed by place identifier (e.g., tract ID
        or parcel ID), represented by x and y coordinate columns
    method : str, {'euclidean', 'greatcircle', 'network'}
        which algorithm to use for calculating pairwise distances
    x : str
        label of the x coordinate column in the DataFrame
    y : str
        label of the y coordinate column in the DataFrame
    earth_radius : numeric
        if method='greatcircle', radius of earth in units in which distance will
        be returned (default is meters)
    return_int : bool
        if method='greatcircle', if True, convert all distances to integers

    Returns
    -------
    df_dist_matrix : pandas DataFrame
        square distance matrix in units of earth_radius
    """

    if not df.index.is_unique:
        raise ValueError('The passed-in DataFrame must have a unique index')

    if method == 'euclidean':
        return euclidean_distance_matrix(df=df, x=x, y=y)
    elif method == 'greatcircle':
        return great_circle_distance_matrix(df=df, x=x, y=y, earth_radius=earth_radius,
                                            return_int=return_int)
    elif method == 'network':
        return network_distance_matrix(df=df)
    else:
        raise ValueError('method argument value must be one of "euclidean", "greatcircle", or "network"')




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




def distance_bands(dist_matrix, distances):
    """
    Iterate through a list, pairwise.

    Parameters
    ----------
    dist_matrix : pandas DataFrame
        a distance matrix with rows and columns indexed by geography ID
    distances : list
        a list of distance band increments

    Returns
    -------
    df : pandas DataFrame
        a DataFrame multi-indexed by geography ID and distance band number, with
        values of arrays of geography IDs with the corresponding distances from
        that ID
    """

    # loop through each row in distance matrix, identifying all tracts within each distance band of the row's tract
    tract_bands = {}
    for _, row in dist_matrix.iterrows():
        tract_bands[row.name] = {}

        # for each distance band
        for i, (dist1, dist2) in enumerate(pairwise(distances)):

            # find all the tracts within this band
            mask = (row >= dist1) & (row < dist2)
            place_ids = row[mask].index.values

            # store value as array of tract IDs keyed by reference tract ID and distance band number
            tract_bands[row.name][i + 1] = place_ids

    # convert tract bands to a dataframe indexed by tract ID and distance band number
    df = pd.DataFrame(tract_bands).T.stack()
    return df
