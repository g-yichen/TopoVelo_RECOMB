import scanpy
import numpy as np
from scipy.sparse import block_diag, csr_matrix
from .scvelo_preprocessing import *
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def count_peak_expression(adata, cluster_key="clusters"):
    # Count the number of genes with peak expression in each cell type.
    def encodeType(cell_types_raw):
        # Use integer to encode the cell types.
        # Each cell type has one unique integer label.
        # Map cell types to integers
        label_dic = {}
        label_dic_rev = {}
        for i, type_ in enumerate(cell_types_raw):
            label_dic[type_] = i
            label_dic_rev[i] = type_

        return label_dic, label_dic_rev
    cell_labels = adata.obs[cluster_key]
    cell_types = np.unique(cell_labels)
    label_dic, label_dic_rev = encodeType(cell_types)
    cell_labels = np.array([label_dic[x] for x in cell_labels])
    n_type = len(cell_types)

    X = np.array(adata.layers["spliced"].A+adata.layers["unspliced"].A)
    peak_expression = np.stack([np.quantile(X[cell_labels == j], 0.9, 0) for j in range(n_type)])
    peak_type = np.argmax(peak_expression, 0)
    peak_hist = np.array([np.sum(peak_type == i) for i in range(n_type)])  # gene count
    peak_val_hist = [peak_expression[:, peak_type == i][i] for i in range(n_type)]  # peak expression
    peak_gene = [np.where(peak_type == i)[0] for i in range(n_type)]  # peak gene index list

    out_peak_count = {}
    out_peak_expr = {}
    out_peak_gene = {}
    for i in range(n_type):
        out_peak_count[label_dic_rev[i]] = peak_hist[i]
        out_peak_expr[label_dic_rev[i]] = peak_val_hist[i]
        out_peak_gene[label_dic_rev[i]] = np.array(peak_gene[i])

    return out_peak_count, out_peak_expr, out_peak_gene


def balanced_gene_selection(adata, n_gene, cluster_key):
    # select the same number of genes for each cell type.
    if n_gene > adata.n_vars:
        return
    cell_labels = adata.obs[cluster_key].to_numpy()
    cell_types = np.unique(cell_labels)
    n_type = len(cell_types)
    count, peak_expr, peak_gene = count_peak_expression(adata, cluster_key)
    length_list = [len(peak_gene[x]) for x in cell_types]
    order_length = np.argsort(length_list)
    k = 0
    s = 0
    while s+length_list[order_length[k]]*(n_type-k) < n_gene:
        s = s+length_list[order_length[k]]
        k = k+1

    gene_list = []
    # Cell types with all peak genes picked
    for i in range(k):
        gene_list.extend(peak_gene[cell_types[order_length[i]]])
    n_gene_per_type = (n_gene - s)//(n_type-k)
    for i in range(k, n_type-1):
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[i]]]))
        gene_list.extend(peak_gene[cell_types[order_length[i]]][gene_idx_order[:n_gene_per_type]])
    if k < n_type-1:
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[-1]]]))
        n_res = n_gene - s - n_gene_per_type*(n_type-k-1)
        gene_list.extend(peak_gene[cell_types[order_length[-1]]][gene_idx_order[:n_res]])
    gene_subsel = np.zeros((adata.n_vars), dtype=bool)
    gene_subsel[np.array(gene_list).astype(int)] = True
    adata._inplace_subset_var(gene_subsel)
    return


def filt_gene_sparsity(adata, thred_u=0.99, thred_s=0.99):
    N, G = adata.n_obs, adata.n_vars
    sparsity_u = np.zeros((G))
    sparsity_s = np.zeros((G))
    for i in tqdm(range(G)):
        sparsity_u[i] = np.sum(adata.layers["unspliced"][:, i].A.squeeze() == 0)/N
        sparsity_s[i] = np.sum(adata.layers["spliced"][:, i].A.squeeze() == 0)/N
    gene_subset = (sparsity_u < thred_u) & (sparsity_s < thred_s)
    print(f"Kept {np.sum(gene_subset)} genes after sparsity filtering")
    adata._inplace_subset_var(gene_subset)


def rank_gene_selection(adata, cluster_key, **kwargs):
    if "cell_types" not in kwargs:
        cell_types = np.unique(adata.obs[cluster_key].to_numpy())
    else:
        cell_types = kwargs["cell_types"]
    use_raw = kwargs["use_raw"] if "use_raw" in kwargs else False
    layer = kwargs["layer"] if "layer" in kwargs else None
    scanpy.tl.rank_genes_groups(adata,
                                groupby=cluster_key,
                                use_raw=use_raw,
                                layer=layer,
                                method='wilcoxon',
                                pts=True)
    min_in_group_fraction = kwargs["min_in_group_fraction"] if "min_in_group_fraction" in kwargs else 0.1
    min_fold_change = kwargs["min_fold_change"] if "min_fold_change" in kwargs else 1.5
    max_out_group_fraction = kwargs["max_out_group_fraction"] if "max_out_group_fraction" in kwargs else 0.5
    compare_abs = kwargs["compare_abs"] if "compare_abs" in kwargs else False
    scanpy.tl.filter_rank_genes_groups(adata,
                                       groupby=cluster_key,
                                       use_raw=False,
                                       min_in_group_fraction=min_in_group_fraction,
                                       min_fold_change=min_fold_change,
                                       max_out_group_fraction=max_out_group_fraction,
                                       compare_abs=compare_abs)
    gene_subset = np.zeros((adata.n_vars), dtype=bool)
    # Build a gene index mapping
    gene_dic = {}
    for i, x in enumerate(adata.var_names):
        gene_dic[x] = i
    gene_set = set()
    for ctype in cell_types:
        names = adata.uns['rank_genes_groups_filtered']['names'][ctype].astype(str)
        adata.uns['rank_genes_groups_filtered']['names'][ctype] = names
        gene_set = gene_set.union(set(names))
    for gene in gene_set:
        if gene != 'nan':
            gene_subset[gene_dic[gene]] = True
    print(f"Picked {len(gene_set)-1} genes")
    adata._inplace_subset_var(gene_subset)
    del adata.uns['rank_genes_groups']['pts']
    del adata.uns['rank_genes_groups']['pts_rest']
    del adata.uns['rank_genes_groups_filtered']['pts']
    del adata.uns['rank_genes_groups_filtered']['pts_rest']


def preprocess(adata,
               n_gene=1000,
               cluster_key="clusters",
               spatial_key=None,
               spatial_smoothing=False,
               tkey=None,
               selection_method="scv",
               min_count_per_cell=None,
               min_genes_expressed=None,
               min_shared_counts=10,
               min_shared_cells=10,
               min_counts_s=None,
               min_cells_s=None,
               max_counts_s=None,
               max_cells_s=None,
               min_counts_u=None,
               min_cells_u=None,
               max_counts_u=None,
               max_cells_u=None,
               npc=30,
               n_neighbors=30,
               n_spatial_neighbors=50,
               genes_retain=None,
               perform_clustering=False,
               resolution=1.0,
               compute_umap=False,
               divide_compartments=False,
               umap_min_dist=0.5,
               keep_raw=True,
               **kwargs):
    """Run the entire preprocessing pipeline using scanpy

    Arguments
    ---------

    adata : :class:`anndata.AnnData`
    n_gene : int, optional
        Number of genes to keep
    cluster_key : str, optional
        Key in adata.obs containing the cell type
    spatial_key : str, optional
        Key in adata.obsm containing the spatial coordinates
        Defaults to None
    spatial_smoothing : bool, optional
        Whether to use the spatial graph to smooth the data.
        Defaults to False.
    tkey : str, optional
        Key in adata.obs containing the capture time
    selection_method : {'scv','balanced'}, optional
        If set to 'balanced', the function will call balanced_gene_selection.
        Otherwise, it uses scanpy to pick highly variable genes.
    min_count_per_cell...max_cells_u : int, optional
        RNA count threshold
    npc : int, optional
        Number of principal components in PCA dimension reduction
    n_neighbors : int, optional
        Number of neighbors in KNN graph
    n_spatial_neighbors : int, optional
        Number of spatial neighbors in spatial KNN graph
    genes_retain : `numpy array` or string list, optional
        By setting genes_retain to a specific list of gene names
        preprocessing will pick these exact genes regardless of their counts and gene selection method.
    perform_clustering : bool, optional
        Whether to perform Leiden clustering
    resolution : float, optional
        Leiden clustering hyperparameter.
    compute_umap : bool, optional
        Whether to compute 2D UMAP
    umap_min_dist : float, optional
        UMAP hyperparameter. Usually is set to less than 1
    """
    # Preprocessing
    # 1. Cell, Gene filtering and data normalization
    n_cell = adata.n_obs
    if min_count_per_cell is None:
        min_count_per_cell = n_gene * 0.5
    if min_genes_expressed is None:
        min_genes_expressed = n_gene // 50
    scanpy.pp.filter_cells(adata, min_counts=min_count_per_cell)
    scanpy.pp.filter_cells(adata, min_genes=min_genes_expressed)
    if n_cell - adata.n_obs > 0:
        print(f"Filtered out {n_cell - adata.n_obs} cells with low counts.")

    if keep_raw:
        gene_names_all = np.array(adata.var_names)
        U_raw = adata.layers["unspliced"]
        S_raw = adata.layers["spliced"]

    if n_gene > 0:
        flavor = kwargs["flavor"] if "flavor" in kwargs else "seurat"
        if selection_method == "balanced":
            print("Balanced gene selection.")
            filter_genes(adata,
                         min_counts=min_counts_s,
                         min_cells=min_cells_s,
                         max_counts=max_counts_s,
                         max_cells=max_cells_s,
                         min_counts_u=min_counts_u,
                         min_cells_u=min_cells_u,
                         max_counts_u=max_counts_u,
                         max_cells_u=max_cells_u,
                         retain_genes=genes_retain)
            balanced_gene_selection(adata, n_gene, cluster_key)
            normalize_per_cell(adata)
        elif selection_method == "wilcoxon":
            print("Marker gene selection using Wilcoxon test.")
            filter_genes(adata,
                         min_counts=min_counts_s,
                         min_cells=min_cells_s,
                         max_counts=max_counts_s,
                         max_cells=max_cells_s,
                         min_counts_u=min_counts_u,
                         min_cells_u=min_cells_u,
                         max_counts_u=max_counts_u,
                         max_cells_u=max_cells_u,
                         retain_genes=genes_retain)
            normalize_per_cell(adata)
            log1p(adata)
            if adata.n_vars > n_gene:
                filter_genes_dispersion(adata,
                                        n_top_genes=n_gene,
                                        retain_genes=genes_retain,
                                        flavor=flavor)
            rank_gene_selection(adata, cluster_key, **kwargs)
        else:
            filter_and_normalize(adata,
                                 min_shared_counts=min_shared_counts,
                                 min_shared_cells=min_shared_cells,
                                 min_counts=min_counts_s,
                                 min_counts_u=min_counts_u,
                                 n_top_genes=n_gene,
                                 retain_genes=genes_retain,
                                 flavor=flavor)
    elif genes_retain is not None:
        gene_subset = np.zeros(adata.n_vars, dtype=bool)
        for i in range(len(genes_retain)):
            indices = np.where(adata.var_names == genes_retain[i])[0]
            if len(indices) == 0:
                continue
            gene_subset[indices[0]] = True
        adata._inplace_subset_var(gene_subset)
        normalize_per_cell(adata)
        log1p(adata)

    # second round of gene filter in case genes in genes_retain don't fulfill
    # minimal count requirement
    if genes_retain is not None:
        filter_genes(adata,
                     min_counts=min_counts_s,
                     min_cells=min_cells_s,
                     max_counts=max_counts_s,
                     max_cells=max_cells_s,
                     min_counts_u=min_counts_u,
                     min_cells_u=min_cells_u,
                     max_counts_u=max_counts_u,
                     max_cells_u=max_cells_u,
                     retain_genes=genes_retain)

    # 2. KNN Averaging
    # remove_duplicate_cells(adata)
    if spatial_smoothing:
        print('Spatial KNN smoothing.')
        moments(adata, n_pcs=npc, n_neighbors=n_spatial_neighbors, method='sklearn', use_rep=spatial_key)
    else:
        moments(adata, n_pcs=npc, n_neighbors=n_neighbors)

    if keep_raw:
        print("Keep raw unspliced/spliced count data.")
        gene_idx = np.array([np.where(gene_names_all == x)[0][0] for x in adata.var_names])
        adata.layers["unspliced"] = U_raw[:, gene_idx].astype(int)
        adata.layers["spliced"] = S_raw[:, gene_idx].astype(int)

    # 3. Obtain cell clusters
    if perform_clustering:
        scanpy.tl.leiden(adata, key_added='clusters', resolution=resolution)
    # 4. Obtain Capture Time (If available)
    if tkey is not None:
        capture_time = adata.obs[tkey].to_numpy()
        if isinstance(capture_time[0], str):
            tprior = np.array([float(x[1:]) for x in capture_time])
        else:
            tprior = capture_time
        tprior = tprior - tprior.min() + 0.01
        adata.obs["tprior"] = tprior

    # 5. Compute Umap coordinates for visulization
    if compute_umap:
        print("Computing UMAP coordinates.")
        if "X_umap" in adata.obsm:
            print("Warning: Overwriting existing UMAP coordinates.")
        scanpy.tl.umap(adata, min_dist=umap_min_dist)

    # 6. Compute the spatial graph
    """
    if spatial_key is not None:
        if not divide_compartments:
            print('Computing spatial KNN graph.')
            X_pos = adata.obsm[spatial_key]
            nn = NearestNeighbors(n_neighbors=n_spatial_neighbors)
            nn.fit(X_pos)
            adata.obsp['spatial_graph'] = nn.kneighbors_graph()
            adata.obsp['connectivities'] = nn.kneighbors_graph(mode='connectivity')
            adata.obsp['distances'] = nn.kneighbors_graph(mode='distance')
        else:
            print('Divide compartments and perform spatial clustering.')
            scanpy.tl.leiden(adata, key_added='compartment', resolution=resolution)
            print(f"{len(np.unique(adata.obs['compartment']))} compartments detected.")
            compartments = adata.obs['compartment'].to_numpy().astype(int)
            blocks = []
            blocks_conn = []
            blocks_dist = []
            for i in range(compartments.max()+1):
                X_pos = adata.obsm[spatial_key][compartments == i]
                nn = NearestNeighbors(n_neighbors=min(n_spatial_neighbors, len(X_pos)-1))
                nn.fit(X_pos)
                blocks.append(nn.kneighbors_graph().toarray())
                blocks_conn.append(nn.kneighbors_graph(mode='connectivity').toarray())
                blocks_dist.append(nn.kneighbors_graph(mode='distance').toarray())
            adata.obsp['spatial_graph'] = block_diag(blocks).tocsr()
            adata.obsp['connectivities'] = block_diag(blocks_conn).tocsr()
            adata.obsp['distances'] = block_diag(blocks_dist).tocsr()
    """
