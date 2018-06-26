"""
Utilities for working with network-based distances and calculating which
network nodes are within various distance bands of other network nodes.
Allows better spatial sampling than Euclidean distances.
"""



import pickle
from itertools import tee

# networkx is a dependency for network-distance calculation
# but let this be an optional dependency for choicemodels
try:
    import networkx as nx
except ImportError as e:
    nx = None


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



def get_reachable_nodes(G, node, dists, weight='length'):
    """
    Get nodes in subgraphs that are reachable within some 
    distance (e.g. spatial, temporal, etc) from some reference
    node. Depends on the networkx package.

    Parameters
    ----------
    G : networkx MultiDiGraph
        the street network graph
    node : int
        the OSM ID of the reference node around which to
        induce subgraphs
    dists : list
        list of distances at which to induce subgraphs around
        the reference node
    weight : string
        the name of the edge attribute to weight shortest
        path distance calculation by

    Returns
    -------
    reachable_nodes : dict
        a dictionary keyed by dist with value of the set of
        node IDs reachable from some reference node within that
        distance
    """

    # check if we were able to import networkx successfully
    if not nx:
        raise ImportError('The networkx package must be installed to use this feature.')

    reachable_nodes = {}
    
    # for each distance value in the list, induce a subgraph of all nodes
    # reachable from the reference node within this distance.
    for dist in dists:
        if dist > 0:
            subgraph = nx.ego_graph(G, node, radius=dist, distance=weight, 
                                    center=True, undirected=False)
            reachable_nodes[dist] = set(subgraph.nodes())
    
    return reachable_nodes



def get_band_nodes(dists, reachable_nodes):
    """
    Get nodes within delimited, annular distance bands from a
    dict of nodes reachable within each distance from some
    reference node.

    Parameters
    ----------
    dists : list
        list of distances to make pairwise to create bands
    reachable_nodes : dict
        a dictionary keyed by dist with value of the set of
        node IDs reachable from some reference node within that
        distance

    Returns
    -------
    band_nodes : dict
        a dictionary keyed by distance-pair bands, with value of
        the set of node IDs reachable from some reference node
        within that distance band
    """

    min_dist = min(dists)
    pairwise_dists = reversed(list(pairwise(sorted(dists))))
    band_nodes = {}

    # for each pair of distances (forming an annular band) in
    # descending order of distance
    for pair in pairwise_dists:
        
        # get the inner and outer distance limits of this band
        inner_limit = min(pair)
        outer_limit = max(pair)
        
        if inner_limit > min_dist:
            # get the set-theoretic difference between the nodes reachable
            # at the outer limit of the ring and the nodes reachable at the
            # inner limit of the ring. this gets the band of nodes.
            band_nodes[pair] = reachable_nodes[outer_limit] - reachable_nodes[inner_limit]
        else:
            # if the band's inner limit is not greater than the minimal
            # distance in the list, then the band is equivalent to all
            # nodes reachable at its outer limit (i.e., it is the inner-
            # most band).
            band_nodes[pair] = reachable_nodes[outer_limit]
    
    return band_nodes



def get_bands(G, dists):
    """
    Create distance bands from a list of distance. Then, for each
    node in graph, find all the nodes that fall between these bands.

    Parameters
    ----------
    G : networkx MultiDiGraph
        the street network graph
    dists : list
        list of distances at which to induce subgraphs around
        the reference node

    Returns
    -------
    bands : dict
        dictionary of dictionaries keyed by graph node : distance
        bands : all graph nodes reachable from the reference node
        within this distance band
    """

    bands = {}
    nodes = list(G.nodes())
    
    # for each node in the graph, first get all reachable nodes
    # at each distance, then convert these into annular bands
    for node in nodes[0:10]:    
    
        reachable_nodes = get_reachable_nodes(G, node, dists)
        bands[node] = get_band_nodes(dists, reachable_nodes)

    return bands


def pickle_bands(bands, filepath, mode='wb',
                 protocol=pickle.HIGHEST_PROTOCOL):
    """
    Pickle a node distance bands dictionary and save it
    to disk.

    Parameters
    ----------
    bands : dict
        dictionary of node distance bands with nodes reachable
        within each of those bands
    filepath : string
        location where to save the pickled dictionary
    mode : string
        file i/o mode
    protocol : int
        protocol for pickler to use

    Returns
    -------
    None
    """

    with open(filepath, mode) as f:
        pickle.dump(bands, f, protocol=protocol)
