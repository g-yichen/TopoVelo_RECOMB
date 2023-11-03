"""
This module modifies the orignal scVelo source code to 
adapt to velocity graph computation across multiple slices in
spatial transcriptomics.

`Reference: <https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_graph.py#L121>`__
"""
import numpy as np
from numpy import ndarray
from typing import Union
import scanpy as sc
from scanpy import Neighbors
from tqdm.autonotebook import trange

from scipy.sparse import issparse, spmatrix, coo_matrix
from umap.umap_ import fuzzy_simplicial_set
from ..scvelo_preprocessing.moments import get_moments


def get_indices_from_csr(conn):
    """TODO."""
    # extracts indices from connectivity matrix, pads with nans
    ixs = np.ones((conn.shape[0], np.max((conn > 0).sum(1)))) * np.nan
    for i in range(ixs.shape[0]):
        cell_indices = conn[i, :].indices
        ixs[i, : len(cell_indices)] = cell_indices
    return ixs


def get_neighs(adata, knn_graph_key="neighbors", mode="distances"):
    if knn_graph_key in adata.uns.keys():
        if mode in adata.uns[knn_graph_key]:
            return adata.uns[knn_graph_key][mode]
        else:
            raise ValueError("The selected mode is not valid.")
    else:
        raise ValueError("The graph key is not valid.")


def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):
    D = dist.copy()
    D.data += 1e-6

    n_counts = sum(D > 0, axis=1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D


def get_n_neighs(adata):
    return adata.uns.get("neighbors", {}).get("params", {}).get("n_neighbors", 0)


def get_iterative_indices(
    indices,
    index,
    n_recurse_neighbors=2,
    max_neighs=None,
):

    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices


def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()


def l2_norm(x: Union[ndarray, spmatrix], axis: int = 1) -> Union[float, ndarray]:
    """Calculate l2 norm along a given axis.

    Arguments
    ---------
    x
        Array to calculate l2 norm of.
    axis
        Axis along which to calculate l2 norm.

    Returns
    -------
    Union[float, ndarray]
        L2 norm along a given axis.
    """
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    elif x.ndim == 1:
        return np.sqrt(np.einsum("i, i -> ", x, x))
    elif axis == 0:
        return np.sqrt(np.einsum("ij, ij -> j", x, x))
    elif axis == 1:
        return np.sqrt(np.einsum("ij, ij -> i", x, x))


def cosine_correlation(dX, Vi):
    dx = dX - dX.mean(-1)[:, None]
    Vi_norm = l2_norm(Vi, axis=0)

    if Vi_norm == 0:
        result = np.zeros(dx.shape[0])
    else:
        result = (
            np.einsum("ij, j", dx, Vi) / (l2_norm(dx, axis=1) * Vi_norm)[None, :]
        )
    return result


def compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """Computes fuzzy simplical set associated with data.

    This is from umap.fuzzy_simplicial_set :cite:p:`McInnes18`.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):  # umap returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = get_csr_from_indices(knn_indices, knn_dists, n_obs, n_neighbors)

    return distances, connectivities.tocsr()


class VelocityGraph:
    """Extension of scVelo VelocityGraph to cross-slice and customized knn."""

    def __init__(
        self,
        adata,
        which_knn="neighbors",
        vkey="velocity",
        xkey="Ms",
        basis=None,
        n_neighbors=None,
        sqrt_transform=None,
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        gene_subset=None,
        report=False,
        compute_uncertainties=None,
        mode_neighbors="distances",
    ):
        # Select genes
        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}_genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        X = np.array(
            adata.layers[xkey].A[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )
        V = np.array(
            adata.layers[vkey].A[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        self.X = np.array(X, dtype=np.float32)
        self.V = np.array(V, dtype=np.float32)
        self.V_raw = np.array(self.V)

        self.sqrt_transform = sqrt_transform
        uns_key = f"{vkey}_params"
        if self.sqrt_transform is None:
            if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
                self.sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis=1)[:, None]

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2
        # use the default transriptomic knn if which_knn is not found
        if which_knn not in adata.uns.keys():
            sc.pp.neighbors(adata)
        if np.min((get_neighs(adata, which_knn, "distances") > 0).sum(1).A1) == 0:
            raise ValueError(
                "Your neighbor graph seems to be corrupted. "
                "Consider recomputing via pp.neighbors."
            )
        if n_neighbors is None or n_neighbors <= get_n_neighs(adata):
            self.indices = get_indices(
                dist=get_neighs(adata, which_knn, "distances"),
                n_neighbors=n_neighbors,
                mode_neighbors=mode_neighbors,
            )[0]
        else:
            print('Number of neighbors exceeds current maximum. Recomputing KNN.')
            if basis is None:
                basis_keys = ["X_pca", "X_tsne", "X_umap"]
                basis = [key for key in basis_keys if key in adata.obsm.keys()][-1]
            elif f"X_{basis}" in adata.obsm.keys():
                basis = f"X_{basis}"

            neighs = Neighbors(adata)
            neighs.compute_neighbors(
                n_neighbors=n_neighbors, use_rep=basis, n_pcs=10
            )
            self.indices = get_indices(
                dist=neighs.distances, mode_neighbors=mode_neighbors
            )[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None
        self.report = report
        self.adata = adata

    def compute_cosines(self):
        n_obs = self.X.shape[0]
        vals, rows, cols, uncertainties = [], [], [], []
        if self.compute_uncertainties:
            moments = get_moments(self.adata, np.sign(self.V_raw), second_order=True)

        for obs_id in trange(self.X.shape[0]):
            if self.V[obs_id].max() != 0 or self.V[obs_id].min() != 0:
                neighs_idx = get_iterative_indices(
                    self.indices, obs_id, self.n_recurse_neighbors, self.max_neighs
                )

                dX = self.X[neighs_idx] - self.X[obs_id, None]  # 60% of runtime
                if self.sqrt_transform:
                    dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                val = cosine_correlation(dX, self.V[obs_id])  # 40% of runtime

                if self.compute_uncertainties:
                    dX /= l2_norm(dX)[:, None]
                    uncertainties.extend(
                        np.nansum(dX**2 * moments[obs_id][None, :], 1)
                    )

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * obs_id)
                cols.extend(neighs_idx)

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )
        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties, rows, cols, shape=(n_obs, n_obs), split_negative=False
            )
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

        return


def velocity_graph(
    data,
    vkey="velocity",
    xkey="Ms",
    which_knn="neighbors",
    basis=None,
    n_neighbors=None,
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    sqrt_transform=None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties=None,
    approx=None,
    mode_neighbors="distances",
    copy=False,
    n_jobs=None,
    backend="loky",
):
    r"""Computes velocity graph based on cosine similarities.

    The cosine similarities are computed between velocities and potential cell state
    transitions, i.e. it measures how well a corresponding change in gene expression
    :math:`\delta_{ij} = x_j - x_i` matches the predicted change according to the
    velocity vector :math:`\nu_i`,

    .. math::
        \pi_{ij} = \cos\angle(\delta_{ij}, \nu_i)
        = \frac{\delta_{ij}^T \nu_i}{\left\lVert\delta_{ij}\right\rVert
        \left\lVert \nu_i \right\rVert}.

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    xkey: `str` (default: `'Ms'`)
        Layer key to extract count data from.
    which_knn: `str` (default: `'neighbors'`)
        Uns key to extract knn graph from.
    basis: `str` (default: `None`)
        Basis / Embedding to use.
    n_neighbors: `int` or `None` (default: None)
        Use fixed number of neighbors or do recursive neighbor search (if `None`).
    n_recurse_neighbors: `int` (default: `None`)
        Number of recursions for neighbors search. Defaults to
        2 if mode_neighbors is 'distances', and 1 if mode_neighbors is 'connectivities'.
    random_neighbors_at_max: `int` or `None` (default: `None`)
        If number of iterative neighbors for an individual cell is higher than this
        threshold, a random selection of such are chosen as reference neighbors.
    sqrt_transform: `bool` (default: `False`)
        Whether to variance-transform the cell states changes
        and velocities before computing cosine similarities.
    gene_subset: `list` of `str`, subset of adata.var_names or `None`(default: `None`)
        Subset of genes to compute velocity graph on exclusively.
    compute_uncertainties: `bool` (default: `None`)
        Whether to compute uncertainties along with cosine correlation.
    approx: `bool` or `None` (default: `None`)
        If True, first 30 pc's are used instead of the full count matrix
    mode_neighbors: 'str' (default: `'distances'`)
        Determines the type of KNN graph used. Options are 'distances' or
        'connectivities'. The latter yields a symmetric graph.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.
    n_jobs: `int` or `None` (default: `None`)
        Number of parallel jobs.
    backend: `str` (default: "loky")
        Backend used for multiprocessing. See :class:`joblib.Parallel` for valid
        options.

    Returns
    -------
    velocity_graph: `.uns`
        sparse matrix with correlations of cell state transitions with velocities
    """
    adata = data.copy() if copy else data
    verify_neighbors(adata)
    if vkey not in adata.layers.keys():
        velocity(adata, vkey=vkey)
    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=tkey,
        basis=basis,
        n_neighbors=n_neighbors,
        approx=approx,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        sqrt_transform=sqrt_transform,
        gene_subset=gene_subset,
        compute_uncertainties=compute_uncertainties,
        report=True,
        mode_neighbors=mode_neighbors,
    )

    if isinstance(basis, str):
        logg.warn(
            f"The velocity graph is computed on {basis} embedding coordinates.\n"
            f"        Consider computing the graph in an unbiased manner \n"
            f"        on full expression space by not specifying basis.\n"
        )

    n_jobs = get_n_jobs(n_jobs=n_jobs)
    logg.info(
        f"computing velocity graph (using {n_jobs}/{os.cpu_count()} cores)", r=True
    )
    vgraph.compute_cosines(n_jobs=n_jobs, backend=backend)

    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg

    if vgraph.uncertainties is not None:
        adata.uns[f"{vkey}_graph_uncertainties"] = vgraph.uncertainties

    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}
    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{vkey}_graph', sparse matrix with cosine correlations (adata.uns)"
    )

    return adata if copy else None