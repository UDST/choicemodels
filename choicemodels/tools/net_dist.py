"""
Utilities for working with network-based distances and calculating which
network nodes are within various distance bands of other network nodes
"""



import networkx as nx
import pickle
from itertools import tee



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



def get_subgraph_nodes(G, node, dists, weight='length'):
    """
    Get nodes in subgraphs some distance from reference node.

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
    subgraph_nodes : dict
        a dictionary keyed by (dist1, dist2) with value of
        set of node IDs in that subgraph
    """

    subgraph_nodes = {}
    
    for dist1, dist2 in pairwise(dists):
        G_sub = nx.ego_graph(G, node, radius=dist2, distance=weight, center=True, undirected=False)
        subgraph_nodes[(dist1, dist2)] = set(G_sub.nodes())
    
    return subgraph_nodes



def get_band_nodes(dists, subgraph_nodes):
    """
    Get nodes in distance bands from subgraph nodes.

    Parameters
    ----------
    dists : list
        list of distances at which to induce subgraphs around
        the reference node
    subgraph_nodes : dict
        a dictionary keyed by (dist1, dist2) with value of
        set of node IDs in that subgraph

    Returns
    -------
    band_nodes : dict
        
    """

    band_nodes = {}
    pairwise_dists = list(pairwise(dists))
    
    for idx1, dist_pair1 in reversed(list(enumerate(pairwise_dists))):

        if idx1 < 1:
            break

        dist_pair2 = pairwise_dists[idx1 - 1]
        band_nodes[dist_pair1] = subgraph_nodes[dist_pair1] - subgraph_nodes[dist_pair2]

    band_nodes[pairwise_dists[0]] = subgraph_nodes[pairwise_dists[0]]
    
    return band_nodes



def get_bands(G, dists):
    """
    Create distance bands from a list of distances then, for each
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
    node_band_nodes : dict
        dictionary of node distance bands with nodes reachable
        within each of those bands
    """

    node_band_nodes = {}
    nodes = list(G.nodes())
    
    for node in nodes[0:10]:    
    
        subgraph_nodes = get_subgraph_nodes(G, node, dists)
        node_band_nodes[node] = get_band_nodes(dists, subgraph_nodes)

    return node_band_nodes


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
