import logging
import math
import torch
from torch import Tensor
from torch.autograd import Function
from scipy.stats import t

import dgl
from dgl.backend import astype
from dgl.base import ALL, is_all
from dgl.heterograph_index import HeteroGraphIndex
from dgl.sparse import _gsddmm, _gspmm

import networkx as nx
import numpy as np
from tqdm import tqdm


def get_batch_id(num_nodes: torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full(
            (num_nodes[i],), i, dtype=torch.long, device=num_nodes.device
        )
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(
    x: torch.Tensor,
    ratio: float,
    batch_id: torch.Tensor,
    num_nodes: torch.Tensor,
):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.

    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full(
        (batch_size * max_num_nodes,), torch.finfo(x.dtype).min
    )
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device)
        + i * max_num_nodes
        for i in range(batch_size)
    ]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


def _neighbor_sort(
    scores: Tensor,
    end_n_ids: Tensor,
    in_degrees: Tensor,
    cum_in_degrees: Tensor,
):
    """Sort edge scores for each node"""
    num_nodes, max_in_degree = in_degrees.size(0), int(in_degrees.max().item())

    # Compute the index for dense score matrix with size (N x D_{max})
    # Note that the end_n_ids here is the end_node tensor in dgl graph,
    # which is not grouped by its node id (i.e. in this form: 0,0,1,1,1,...,N,N).
    # Thus here we first sort the end_node tensor to make it easier to compute
    # indexs in dense edge score matrix. Since we will need the original order
    # for following gspmm and gsddmm operations, we also keep the reverse mapping
    # (the reverse_perm) here.
    end_n_ids, perm = torch.sort(end_n_ids)
    scores = scores[perm]
    _, reverse_perm = torch.sort(perm)

    index = torch.arange(
        end_n_ids.size(0), dtype=torch.long, device=scores.device
    )
    index = (index - cum_in_degrees[end_n_ids]) + (end_n_ids * max_in_degree)
    index = index.long()

    dense_scores = scores.new_full(
        (num_nodes * max_in_degree,), torch.finfo(scores.dtype).min
    )
    dense_scores[index] = scores
    dense_scores = dense_scores.view(num_nodes, max_in_degree)

    sorted_dense_scores, dense_reverse_perm = dense_scores.sort(
        dim=-1, descending=True
    )
    _, dense_reverse_perm = torch.sort(dense_reverse_perm, dim=-1)
    dense_reverse_perm = dense_reverse_perm + cum_in_degrees.view(-1, 1)
    dense_reverse_perm = dense_reverse_perm.view(-1)
    cumsum_sorted_dense_scores = sorted_dense_scores.cumsum(dim=-1).view(-1)
    sorted_dense_scores = sorted_dense_scores.view(-1)
    arange_vec = torch.arange(
        1, max_in_degree + 1, dtype=torch.long, device=end_n_ids.device
    )
    arange_vec = torch.repeat_interleave(
        arange_vec.view(1, -1), num_nodes, dim=0
    ).view(-1)

    valid_mask = sorted_dense_scores != torch.finfo(scores.dtype).min
    sorted_scores = sorted_dense_scores[valid_mask]
    cumsum_sorted_scores = cumsum_sorted_dense_scores[valid_mask]
    arange_vec = arange_vec[valid_mask]
    dense_reverse_perm = dense_reverse_perm[valid_mask].long()

    return (
        sorted_scores,
        cumsum_sorted_scores,
        arange_vec,
        reverse_perm,
        dense_reverse_perm,
    )


def _threshold_and_support_graph(
    gidx: HeteroGraphIndex, scores: Tensor, end_n_ids: Tensor
):
    """Find the threshold for each node and its edges"""
    in_degrees = _gspmm(gidx, "copy_rhs", "sum", None, torch.ones_like(scores))[
        0
    ]
    cum_in_degrees = torch.cat(
        [in_degrees.new_zeros(1), in_degrees.cumsum(dim=0)[:-1]], dim=0
    )

    # perform sort on edges for each node
    (
        sorted_scores,
        cumsum_scores,
        rhos,
        reverse_perm,
        dense_reverse_perm,
    ) = _neighbor_sort(scores, end_n_ids, in_degrees, cum_in_degrees)
    cumsum_scores = cumsum_scores - 1.0
    support = rhos * sorted_scores > cumsum_scores
    support = support[dense_reverse_perm]  # from sorted order to unsorted order
    support = support[reverse_perm]  # from src-dst order to eid order

    support_size = _gspmm(gidx, "copy_rhs", "sum", None, support.float())[0]
    support_size = support_size.long()
    idx = support_size + cum_in_degrees - 1

    # mask invalid index, for example, if batch is not start from 0 or not continuous, it may result in negative index
    mask = idx < 0
    idx[mask] = 0
    tau = cumsum_scores.gather(0, idx.long())
    tau /= support_size.to(scores.dtype)

    return tau, support_size


class EdgeSparsemaxFunction(Function):
    r"""
    Description
    -----------
    Pytorch Auto-Grad Function for edge sparsemax.

    We define this auto-grad function here since
    sparsemax involves sort and select, which are
    not derivative.
    """

    @staticmethod
    def forward(
        ctx,
        gidx: HeteroGraphIndex,
        scores: Tensor,
        eids: Tensor,
        end_n_ids: Tensor,
        norm_by: str,
    ):
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()

        # use feat - max(feat) for numerical stability.
        scores = scores.float()
        scores_max = _gspmm(gidx, "copy_rhs", "max", None, scores)[0]
        scores = _gsddmm(gidx, "sub", scores, scores_max, "e", "v")

        # find threshold for each node and perform ReLU(u-t(u)) operation.
        tau, supp_size = _threshold_and_support_graph(gidx, scores, end_n_ids)
        out = torch.clamp(_gsddmm(gidx, "sub", scores, tau, "e", "v"), min=0)
        ctx.backward_cache = gidx
        ctx.save_for_backward(supp_size, out)
        torch.cuda.empty_cache()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gidx = ctx.backward_cache
        supp_size, out = ctx.saved_tensors
        grad_in = grad_out.clone()

        # grad for ReLU
        grad_in[out == 0] = 0

        # dL/dv_i = dL/do_i - 1/k \sum_{j=1}^k dL/do_j
        v_hat = _gspmm(gidx, "copy_rhs", "sum", None, grad_in)[
            0
        ] / supp_size.to(out.dtype)
        grad_in_modify = _gsddmm(gidx, "sub", grad_in, v_hat, "e", "v")
        grad_in = torch.where(out != 0, grad_in_modify, grad_in)
        del gidx
        torch.cuda.empty_cache()

        return None, grad_in, None, None, None


def edge_sparsemax(graph: dgl.DGLGraph, logits, eids=ALL, norm_by="dst"):
    r"""
    Description
    -----------
    Compute edge sparsemax. For a node :math:`i`, edge sparsemax is an operation that computes

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge sparsemax on.
    logits : torch.Tensor
        The input edge feature.
    eids : torch.Tensor or ALL, optional
        A tensor of edge index on which to apply edge sparsemax. If ALL, apply edge
        sparsemax on all edges in the graph. Default: ALL.
    norm_by : str, could be 'src' or 'dst'
        Normalized by source nodes of destination nodes. Default: `dst`.

    Returns
    -------
    Tensor
        Sparsemax value.
    """
    # we get edge index tensors here since it is
    # hard to get edge index with HeteroGraphIndex
    # object without other information like edge_type.
    row, col = graph.all_edges(order="eid")
    assert norm_by in ["dst", "src"]
    end_n_ids = col if norm_by == "dst" else row
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
        end_n_ids = end_n_ids[eids]
    return EdgeSparsemaxFunction.apply(
        graph._graph, logits, eids, end_n_ids, norm_by
    )


class EdgeSparsemax(torch.nn.Module):
    r"""
    Description
    -----------
    Compute edge sparsemax. For a node :math:`i`, edge sparsemax is an operation that computes

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge sparsemax on.
    logits : torch.Tensor
        The input edge feature.
    eids : torch.Tensor or ALL, optional
        A tensor of edge index on which to apply edge sparsemax. If ALL, apply edge
        sparsemax on all edges in the graph. Default: ALL.
    norm_by : str, could be 'src' or 'dst'
        Normalized by source nodes of destination nodes. Default: `dst`.

    NOTE: currently only homogeneous graphs are supported.

    Returns
    -------
    Tensor
        Sparsemax value.
    """

    def __init__(self):
        super(EdgeSparsemax, self).__init__()

    def forward(self, graph, logits, eids=ALL, norm_by="dst"):
        return edge_sparsemax(graph, logits, eids, norm_by)


def net_graph(features, couples, edg_feat):
    """
    Create a networkx graph

    Parameters
    ----------
    features : list of np.arrays
        list of features per node
    couples : list of tuples
        list of the related nodes
    edg_feat : list of np.arrays
        list of features per node
    """
    G = nx.Graph()
    for i in range(len(couples)):
        # Keep only edges not based on distance
        if sum(edg_feat[i, 2:]) > 0:
            [l, m] = couples[i]
            G.add_edge(l, m)
    for node in range(len(features)):
        G.add_node(node, feature=features[node])
    return G


def train(model: torch.nn.Module, optimizer, loss_metric, trainloader, device):
    """
    Train the given model with the input dataloader

    Parameters
    ----------
    model : torch model
    optimizer : torch optimizer
    loss_metric : torch loss
    trainloader : GraphDataLoader object
    device : device to train the model (cpu or cuda)

    Returns
    -------
    (Train loss, train accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    num_batches = len(trainloader)
    num_graph = 0
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        num_graph += len(batch_labels)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feature"].to(dtype=torch.float64))
        loss = loss_metric(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += int(out.argmax(dim=-1).eq(batch_labels).sum())
        out.detach()
        loss.detach()

    return total_loss / num_batches, total_correct / num_graph


@torch.no_grad()
def test(model: torch.nn.Module, loss_metric, loader, device):
    """
    Test the given model on the input dataloader

    Parameters
    ----------
    model : torch model
    loss_metric : torch loss
    loader : GraphDataLoader object
    device : device to train the model (cpu or cuda)

    Returns
    -------
    (Test loss, test accuracy)
    """
    model.eval()
    correct = 0.0
    loss = 0.0
    num_batches = len(loader)
    num_graphs = 0
    for batch in tqdm(loader):
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feature"].to(dtype=torch.float32))
        pred = out.argmax(dim=1)
        loss += loss_metric(out, batch_labels).item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_batches


@torch.no_grad()
def extract_prediction(model: torch.nn.Module, loader, device):
    """
    Get the predictions of the model on the input loade given

    Parameters
    ----------
    model : torch model
    trainloader : GraphDataLoader object
    device : device to train the model (cpu or cuda)

    Returns
    -------
    Predictions (np.array)
    """
    model.eval()
    predictions = list()
    for batch in tqdm(loader):
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feature"].to(dtype=torch.float32))
        predictions.append(out.cpu().numpy())
    return np.concatenate(predictions, axis=0)