import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from scipy.stats import norm as normal
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Cellwise Velocity Embedding
def _get_spatial_candidates(coord1, coord2, radius=0.05):
    # normalization
    coord1_ = coord1 - np.min(coord1, 0)
    coord2_ = coord2 - np.min(coord2, 0)
    coord1_ = coord1_ / np.max(coord1_, 0)
    coord2_ = coord2_ / np.max(coord2_, 0)

    # Build an epsilon ball tree
    bt = BallTree(coord2_)
    # Find all cells within the radius
    return bt.query_radius(coord1_, radius, return_distance=False)

def get_spatial_neighbors(adata,
                          basis,
                          spatial_key,
                          slice_key,
                          n_neighbors=30,
                          radius=0.1):
    slice_labels = adata.obs[slice_key].to_numpy()
    slices = np.sort(np.unique(slice_labels))
    if f'X_{basis}' not in adata.obsm:
        raise KeyError
    X = adata.obsm[f'X_{basis}']
    X_spatial = adata.obsm[spatial_key]
    nbs, dist = np.empty((adata.n_obs, n_neighbors)), np.empty((adata.n_obs, n_neighbors))
    for i in range(len(slices) - 1):
        slice_1 = np.where(slice_labels == slices[i])
        slice_2 = np.where(slice_labels == slices[i+1])
        candidates = _get_spatial_candidates(X_spatial[slice_1],
                                             X_spatial[slice_2],
                                             radius)
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X[slice_2][candidates])
        for idx in slice_1:
            dist_i, ind_i = knn.kneighbors(X[idx:idx+1])
            dist[idx] = dist_i[0]
            nbs[idx] = ind_i[0]
    return nbs, dist


def transition_matrix(
    adata,
    vkey="velocity",
    basis=None,
    backward=False,
    self_transitions=True,
    scale=10,
    perc=None,
    threshold=None,
    use_negative_cosines=False,
    weight_diffusion=0,
    scale_diffusion=1,
    weight_indirect_neighbors=None,
    n_neighbors=None,
    vgraph=None,
    basis_constraint=None,
):
    r"""Adapted from scVelo `reference: <https://github.com/theislab/scvelo/blob/master/scvelo/tools/transition_matrix.py#L14>`__
    Computes cell-to-cell transition probabilities.

    .. math::
        \tilde \pi_{ij} = \frac1{z_i} \exp( \pi_{ij} / \sigma),

    from the velocity graph :math:`\pi_{ij}`, with row-normalization :math:`z_i` and
    kernel width :math:`\sigma` (scale parameter :math:`\lambda = \sigma^{-1}`).

    Alternatively, use :func:`cellrank.tl.transition_matrix` to account for uncertainty
    in the velocity estimates.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    basis: `str` or `None` (default: `None`)
        Restrict transition to embedding if specified
    backward: `bool` (default: `False`)
        Whether to use the transition matrix to
        push forward (`False`) or to pull backward (`True`)
    self_transitions: `bool` (default: `True`)
        Allow transitions from one node to itself.
    scale: `float` (default: 10)
        Scale parameter of gaussian kernel.
    perc: `float` between `0` and `100` or `None` (default: `None`)
        Determines threshold of transitions to include.
    use_negative_cosines: `bool` (default: `False`)
        If True, negatively similar transitions are taken into account.
    weight_diffusion: `float` (default: 0)
        Relative weight to be given to diffusion kernel (Brownian motion)
    scale_diffusion: `float` (default: 1)
        Scale of diffusion kernel.
    weight_indirect_neighbors: `float` between `0` and `1` or `None` (default: `None`)
        Weight to be assigned to indirect neighbors (i.e. neighbors of higher degrees).
    n_neighbors:`int` (default: None)
        Number of nearest neighbors to consider around each cell.
    vgraph: csr matrix or `None` (default: `None`)
        Velocity graph representation to use instead of adata.uns[f'{vkey}_graph'].

    Returns
    -------
    Returns sparse matrix with transition probabilities.
    """
    if f"{vkey}_graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.velocity_graph` first to compute cosine correlations."
        )

    graph_neg = None
    if vgraph is not None:
        graph = vgraph.copy()
    else:
        if hasattr(adata, "obsp") and f"{vkey}_graph" in adata.obsp.keys():
            graph = csr_matrix(adata.obsp[f"{vkey}_graph"]).copy()
            if f"{vkey}_graph_neg" in adata.obsp.keys():
                graph_neg = adata.obsp[f"{vkey}_graph_neg"]
        else:
            graph = csr_matrix(adata.uns[f"{vkey}_graph"]).copy()
            if f"{vkey}_graph_neg" in adata.uns.keys():
                graph_neg = adata.uns[f"{vkey}_graph_neg"]

    if basis_constraint is not None and f"X_{basis_constraint}" in adata.obsm.keys():
        from sklearn.neighbors import NearestNeighbors

        neighs = NearestNeighbors(n_neighbors=100)
        neighs.fit(adata.obsm[f"X_{basis_constraint}"])
        basis_graph = neighs.kneighbors_graph(mode="connectivity") > 0
        graph = graph.multiply(basis_graph)

    if self_transitions:
        confidence = graph.max(1).A.flatten()
        ub = np.percentile(confidence, 98)
        self_prob = np.clip(ub - confidence, 0, 1)
        graph.setdiag(self_prob)

    T = np.expm1(graph * scale)  # equivalent to np.exp(graph.A * scale) - 1
    if graph_neg is not None:
        graph_neg = adata.uns[f"{vkey}_graph_neg"]
        if use_negative_cosines:
            T -= np.expm1(-graph_neg * scale)
        else:
            T += np.expm1(graph_neg * scale)
            T.data += 1

    # weight direct and indirect (recursed) neighbors
    if weight_indirect_neighbors is not None and weight_indirect_neighbors < 1:
        direct_neighbors = get_neighs(adata, "distances") > 0
        direct_neighbors.setdiag(1)
        w = weight_indirect_neighbors
        T = w * T + (1 - w) * direct_neighbors.multiply(T)

    if n_neighbors is not None:
        T = T.multiply(
            get_connectivities(
                adata, mode="distances", n_neighbors=n_neighbors, recurse_neighbors=True
            )
        )

    if perc is not None or threshold is not None:
        if threshold is None:
            threshold = np.percentile(T.data, perc)
        T.data[T.data < threshold] = 0
        T.eliminate_zeros()

    if backward:
        T = T.T
    T = normalize(T)

    if f"X_{basis}" in adata.obsm.keys():
        dists_emb = (T > 0).multiply(squareform(pdist(adata.obsm[f"X_{basis}"])))
        scale_diffusion *= dists_emb.data.mean()

        diffusion_kernel = dists_emb.copy()
        diffusion_kernel.data = np.exp(
            -0.5 * dists_emb.data**2 / scale_diffusion**2
        )
        T = T.multiply(diffusion_kernel)  # combine velocity kernel & diffusion kernel

        if 0 < weight_diffusion < 1:  # add diffusion kernel (Brownian motion - like)
            diffusion_kernel.data = np.exp(
                -0.5 * dists_emb.data**2 / (scale_diffusion / 2) ** 2
            )
            T = (1 - weight_diffusion) * T + weight_diffusion * diffusion_kernel

        T = normalize(T)

    return T


def velocity_embedding(
    data,
    basis=None,
    vkey="velocity",
    scale=10,
    self_transitions=True,
    use_negative_cosines=True,
    direct_pca_projection=None,
    retain_scale=False,
    autoscale=True,
    all_comps=True,
    T=None,
    copy=False,
):
    r"""Adapted from scVelo `reference: <https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_embedding.py#L32-L202>`__
    Projects the single cell velocities into any embedding.

    Given normalized difference of the embedding positions

    .. math::
        \tilde \delta_{ij} = \frac{x_j-x_i}{\left\lVert x_j-x_i \right\rVert},

    the projections are obtained as expected displacements with respect to the
    transition matrix :math:`\tilde \pi_{ij}` as

    .. math::
        \tilde \nu_i = E_{\tilde \pi_{i\cdot}} [\tilde \delta_{i \cdot}]
        = \sum_{j \neq i} \left( \tilde \pi_{ij} - \frac1n \right) \tilde
        \delta_{ij}.


    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    basis: `str` (default: `'tsne'`)
        Which embedding to use.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    scale: `int` (default: 10)
        Scale parameter of gaussian kernel for transition matrix.
    self_transitions: `bool` (default: `True`)
        Whether to allow self transitions, based on the confidences of transitioning to
        neighboring cells.
    use_negative_cosines: `bool` (default: `True`)
        Whether to project cell-to-cell transitions with negative cosines into
        negative/opposite direction.
    direct_pca_projection: `bool` (default: `None`)
        Whether to directly project the velocities into PCA space,
        thus skipping the velocity graph.
    retain_scale: `bool` (default: `False`)
        Whether to retain scale from high dimensional space in embedding.
    autoscale: `bool` (default: `True`)
        Whether to scale the embedded velocities by a scalar multiplier,
        which simply ensures that the arrows in the embedding are properly scaled.
    all_comps: `bool` (default: `True`)
        Whether to compute the velocities on all embedding components.
    T: `csr_matrix` (default: `None`)
        Allows the user to directly pass a transition matrix.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.

    Returns
    -------
    velocity_umap: `.obsm`
        coordinates of velocity projection on embedding (e.g., basis='umap')
    """
    adata = data.copy() if copy else data

    if basis is None:
        keys = [
            key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()
        ]
        if len(keys) > 0:
            basis = "pca" if direct_pca_projection else keys[-1]
        else:
            raise ValueError("No basis specified")

    if f"X_{basis}" not in adata.obsm_keys():
        raise ValueError("You need to compute the embedding first.")

    if direct_pca_projection and "pca" in basis:
        print(
            "Directly projecting velocities into PCA space is for exploratory analysis "
            "on principal components.\n"
            "         It does not reflect the actual velocity field from high "
            "dimensional gene expression space.\n"
            "         To visualize velocities, consider applying "
            "`direct_pca_projection=False`.\n"
        )

    print("computing velocity embedding")

    V = np.array(adata.layers[vkey])
    vgenes = np.ones(adata.n_vars, dtype=bool)
    if f"{vkey}_genes" in adata.var.keys():
        vgenes &= np.array(adata.var[f"{vkey}_genes"], dtype=bool)
    vgenes &= ~np.isnan(V.sum(0))
    V = V[:, vgenes]

    if direct_pca_projection and "pca" in basis:
        PCs = adata.varm["PCs"] if all_comps else adata.varm["PCs"][:, :2]
        PCs = PCs[vgenes]

        X_emb = adata.obsm[f"X_{basis}"]
        V_emb = (V - V.mean(0)).dot(PCs)

    else:
        X_emb = (
            adata.obsm[f"X_{basis}"] if all_comps else adata.obsm[f"X_{basis}"][:, :2]
        )
        V_emb = np.zeros(X_emb.shape)

        T = (
            transition_matrix(
                adata,
                vkey=vkey,
                scale=scale,
                self_transitions=self_transitions,
                use_negative_cosines=use_negative_cosines,
            )
            if T is None
            else T
        )
        T.setdiag(0)
        T.eliminate_zeros()

        densify = adata.n_obs < 1e4
        TA = T.A if densify else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(adata.n_obs):
                indices = T[i].indices
                dX = X_emb[indices] - X_emb[i, None]  # shape (n_neighbors, 2)
                if not retain_scale:
                    dX /= l2_norm(dX)[:, None]
                dX[np.isnan(dX)] = 0  # zero diff in a steady-state
                probs = TA[i, indices] if densify else T[i].data
                V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)

        if retain_scale:
            X = (
                adata.layers["Ms"]
                if "Ms" in adata.layers.keys()
                else adata.layers["spliced"]
            )
            delta = T.dot(X[:, vgenes]) - X[:, vgenes]
            if issparse(delta):
                delta = delta.A
            cos_proj = (V * delta).sum(1) / l2_norm(delta)
            V_emb *= np.clip(cos_proj[:, None] * 10, 0, 1)

    if autoscale:
        V_emb /= 3 * quiver_autoscale(X_emb, V_emb)

    if f"{vkey}_params" in adata.uns.keys():
        adata.uns[f"{vkey}_params"]["embeddings"] = (
            []
            if "embeddings" not in adata.uns[f"{vkey}_params"]
            else list(adata.uns[f"{vkey}_params"]["embeddings"])
        )
        adata.uns[f"{vkey}_params"]["embeddings"].extend([basis])

    vkey += f"_{basis}"
    adata.obsm[vkey] = V_emb

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added\n" f"    '{vkey}', embedded velocity vectors (adata.obsm)")

    return adata if copy else None


# Velocity Embedding on a 3D grid
def quiver_autoscale(X_emb, V_emb):
    """From scVelo. 
    `reference: <https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_embedding.py#L13>`__"""

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = plt.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


def compute_velocity_on_grid_2d(
    X_emb,
    V_emb,
    X_grid,
    grs,
    smooth=None,
    n_neighbors=None
):
    """From scVelo. 
    `reference: <https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_grid.py#L28>`__
    """

    # estimate grid velocities
    n_obs, n_dim = X_emb.shape
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    # Find knn of each grid point from all cells
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    # Estimate the std of knn distance distribution based on grid distance
    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    # Velocity on grid is a weighted sum of cell velocity,
    # where the weight is pdf of distance to that cell
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    # Used for velocity stream scaling
    length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
    return V_grid, weight, length


def _trim_velocity_grid(
    X_grid_2d,
    V_grid,
    length,
    p_mass,
    min_mass,
    adjust_for_stream=False,
    cutoff_perc=None
):
    _V_grid = V_grid
    if adjust_for_stream:
        _X_grid = np.stack([np.unique(X_grid_2d[:, 0]),
                            np.unique(X_grid_2d[:, 1])])
        ns = int(np.sqrt(len(_V_grid[:, 0])))
        _V_grid = _V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((_V_grid**2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(_V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5

        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        _V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 5
        # _X_grid, _V_grid = X_grid_2d[p_mass > min_mass], _V_grid[p_mass > min_mass]
        _X_grid = X_grid_2d
        _V_grid[p_mass < min_mass] = 0
    return _X_grid, _V_grid


def _get_mid_v(v):
    return np.mean(np.abs(v))


def interpolate_velocity_on_grid(
    X_grid_2d,
    V_slice,
    weights,
    length_slices,
    density=1,
    min_mass=None,
    adjust_for_stream=False,
    cutoff_perc=None,
):
    """Given multiple slices of 2D grids (same coordinates),
    the function interpolates the velocity on grid points
    between the slices.

    Args:
        X_grid_2d (:class:`numpy.ndarray`):
            2D grid
        V_slice (list[:class:`numpy.ndarray`]):
            list of velocity in each slice of 2D grid
        weights (:class:`numpy.ndarray`):
            Weight of each grid point
        length_slices (list[float]):
            List of velocity length in each each slice
        density (float, optional):
            Scaling factor of grid point density. Defaults to None.
        smooth (float, optional):
            Scaling factor of Gaussian std in weight computation. Defaults to None.
        n_neighbors (int, optional):
            Number of neighbors in weight computation. Defaults to None.
        min_mass (float, optional):
            Minimum pdf value to filter out grid points. Defaults to None.
        adjust_for_stream (bool, optional):
            Whether to adjust the arrow length. Defaults to False.
        cutoff_perc (float, optional):
            Used for velocity stream. Defaults to None.
    """
    V_grid = []
    X_grid = []
    z_grid_size = int(5*density)
    if min_mass is None:
        min_mass = 1

    for i in range(len(V_slice)-1):
        for j in range(z_grid_size):
            # k = np.exp(-(j/z_grid_size*2.0)**2)
            k = j/z_grid_size
            weight = weights[i]*k + weights[i+1]*(1-k)
            p_mass = weight.sum(1)
            _V_grid = V_slice[i]*k + V_slice[i+1]*(1-k)
            _V_grid = np.concatenate([_V_grid, np.ones((_V_grid.shape[0], 1))*_get_mid_v(_V_grid)], 1)
            _length_slice = length_slices[i]*k + length_slices[i+1]*(1-k)
            _X_grid, _V_grid = _trim_velocity_grid(X_grid_2d,
                                                   _V_grid,
                                                   _length_slice,
                                                   p_mass,
                                                   min_mass,
                                                   adjust_for_stream,
                                                   cutoff_perc)
            X_grid.append(np.concatenate([_X_grid, np.ones((_X_grid.shape[0], 1))*(i+k)], 1))
            V_grid.append(_V_grid)
    # The last slice
    _X_grid, _V_grid = _trim_velocity_grid(X_grid_2d,
                                           np.concatenate([V_slice[-1],
                                                           np.ones((len(V_slice[-1]), 1))*_get_mid_v(V_slice[-1])], 1),
                                           length_slices[-1],
                                           p_mass,
                                           min_mass,
                                           adjust_for_stream,
                                           cutoff_perc)
    X_grid.append(np.concatenate([_X_grid, np.ones((_X_grid.shape[0], 1))*(len(V_slice)-1)], 1))
    V_grid.append(_V_grid)

    return X_grid, V_grid


def compute_velocity_on_grid_3d(
    X_emb,
    V_emb,
    slice_labels,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    autoscale=True,
    adjust_for_stream=False,
    cutoff_perc=None,
    angle=(15, 45),
    d_init=10
):
    slices = np.sort(np.unique(slice_labels))

    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare a grid common to all slices
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(10 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid_2d = np.vstack([i.flat for i in meshes_tuple]).T
    if n_neighbors is None:
        n_neighbors = max(10, np.min([int(np.sum(slice_labels == s) / 30) for s in slices]))

    V_grid_slices = []
    weight_slices = []
    v_length_slices = []
    for i in range(len(slices)):
        cur_slice = np.where(slice_labels == slices[i])
        V_grid_i, weight, v_length = compute_velocity_on_grid_2d(
                                                                    X_emb[cur_slice],
                                                                    V_emb[cur_slice],
                                                                    X_grid_2d,
                                                                    grs,
                                                                    smooth,
                                                                    n_neighbors
                                                                )
        V_grid_slices.append(V_grid_i)
        weight_slices.append(weight)
        v_length_slices.append(v_length)

    # Interpolation
    X_grid, V_grid = interpolate_velocity_on_grid(X_grid_2d,
                                                  V_grid_slices,
                                                  weight_slices,
                                                  v_length_slices,
                                                  density,
                                                  min_mass,
                                                  adjust_for_stream,
                                                  cutoff_perc)
    # generate 3D grid, order: y, z, x
    grs = []
    m, M = np.min(X_emb[:, 1]), np.max(X_emb[:, 1])
    m = m - 0.01 * np.abs(M - m)
    M = M + 0.01 * np.abs(M - m)
    gr = np.linspace(m, M, int(10 * density))
    grs.append(gr)
    grs.append(np.linspace(0, len(slices)-1, len(X_grid)))
    m, M = np.min(X_emb[:, 0]), np.max(X_emb[:, 0])
    m = m - 0.01 * np.abs(M - m)
    M = M + 0.01 * np.abs(M - m)
    gr = np.linspace(m, M, int(10 * density))
    grs.append(gr)
    
    yg, zg, xg = np.meshgrid(*grs)
    X_grid = np.stack([xg.flatten(), yg.flatten(), zg.flatten()]).T
    return X_grid, V_grid, X_grid_2d, len(X_grid)//len(V_grid)