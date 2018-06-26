"""
Utilities for working with network-based distances and calculating which
network nodes are within various distance bands of other network nodes
"""




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