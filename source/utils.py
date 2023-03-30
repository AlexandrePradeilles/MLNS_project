"""
You will find functions to load the Altegrad data and compute cross entropy
"""
import scipy.sparse as sp
import numpy as np


def load_data(): 
    """
    Function that loads graphs data
    """  
    graph_indicator = np.loadtxt("../data/graph_indicator.txt", dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("../data/edgelist.txt", dtype=np.int64, delimiter=",")
    edges_inv = np.vstack((edges[:,1], edges[:,0]))
    edges = np.vstack((edges, edges_inv.T))
    s = edges[:,0]*graph_indicator.size + edges[:,1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort,:]
    edges,idx_unique =  np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    x = np.loadtxt("../data/node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt("../data/edge_attributes.txt", delimiter=",")
    edge_attr = np.vstack((edge_attr,edge_attr))
    edge_attr = edge_attr[idx_sort,:]
    edge_attr = edge_attr[idx_unique,:]
    
    adj = []
    features = []
    edge_features = []
    edges_couples = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
        couples = edges[idx_m:idx_m+adj[i].nnz,:]
        couples = couples - np.min(couples)
        edges_couples.append(couples)
        features.append(x[idx_n:idx_n+graph_size[i],:])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, features, edge_features, edges_couples


def cross_entropy(predictions, targets):
    """Compute the cross entropy
    """
    N = predictions.shape[0]
    ce = 0
    for i in range(N):
        ce += - np.log(predictions[i, targets[i]]+10**(-30))
    return ce / N
