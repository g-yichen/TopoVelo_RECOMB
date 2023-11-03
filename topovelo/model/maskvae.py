import numpy as np
import os
import torch
import torch.nn as nn
from .vae import VAE
from .model_util import find_dirichlet_param, get_gene_index
from .training_data import SCGraphData
import time
from ..plotting import plot_sig, plot_time
from ..plotting import plot_train_loss, plot_test_loss
from .model_util import convert_time, get_gene_index
from .model_util import pred_su, ode_numpy, knnx0_index, get_x0
from .model_util import elbo_collapsed_categorical
from .model_util import assign_gene_mode, find_dirichlet_param
from .model_util import get_cell_scale, get_dispersion
from .transition_graph import encode_type


class MaskVAE(VAE):
    """Masked Graph VAE Model
    """
    def __init__(self,
                 adata,
                 tmax,
                 dim_z,
                 dim_cond=0,
                 device='cpu',
                 hidden_size=(500, 250, 500),
                 full_vb=False,
                 discrete=False,
                 graph_decoder=False,
                 attention=False,
                 init_method="steady",
                 init_key=None,
                 tprior=None,
                 init_ton_zero=True,
                 filter_gene=False,
                 count_distribution="Poisson",
                 std_z_prior=0.01,
                 checkpoints=[None, None],
                 rate_prior={
                     'alpha': (0.0, 1.0),
                     'beta': (0.0, 0.5),
                     'gamma': (0.0, 0.5)
                 },
                 **kwargs):
        """Masked TopoVelo Model

        Arguments
        ---------

        adata : :class:`anndata.AnnData`
            AnnData object containing all relevant data and meta-data
        tmax : float
            Time range.
            This is used to restrict the cell time within certain range. In the
            case of a Gaussian model without capture time, tmax/2 will be the mean of prior.
            If capture time is provided, then they are scaled to match a range of tmax.
            In the case of a uniform model, tmax is strictly the maximum time.
        dim_z : int
            Dimension of the latent cell state
        dim_cond : int, optional
            Dimension of additional information for the conditional VAE.
            Set to zero by default, equivalent to a VAE.
            This feature is not stable now.
        device : {'gpu','cpu'}, optional
            Training device
        hidden_size : tuple of int, optional
            Width of the hidden layers. Should be a tuple of the form
            (encoder layer 1, encoder layer 2, decoder layer 1, decoder layer 2)
        full_vb : bool, optional
            Enable the full variational Bayes
        discrete : bool, optional
            Enable the discrete count model
        init_method : {'random', 'tprior', 'steady}, optional
            Initialization method.
            Should choose from
            (1) random: random initialization
            (2) tprior: use the capture time to estimate rate parameters. Cell time will be
                        randomly sampled with the capture time as the mean. The variance can
                        be controlled by changing 'time_overlap' in config.
            (3) steady: use the steady-state model to estimate gamma, alpha and assume beta = 1.
                        After this, a global cell time is estimated by taking the quantile over
                        all local times. Finally, rate parameters are reinitialized using the
                        global cell time.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        init_ton_zero : bool, optional
            Whether to add a non-zero switch-on time for each gene.
            It's set to True if there's no capture time.
        filter_gene : bool, optional
            Whether to remove non-velocity genes
        count_distriution : {'auto', 'Poisson', 'NB'}, optional
            Count distribution, effective only when discrete=True
            The current version only assumes Poisson or negative binomial distributions.
            When set to 'auto', the program determines a proper one based on over dispersion
        std_z_prior : float, optional
            Standard deviation of the prior (isotropical Gaussian) of cell state.
        checkpoints : list of 2 strings, optional
            Contains the path to saved encoder and decoder models.
            Should be a .pt file.
        rate_prior : dict, optional
            Prior distribution of rate parameters.
            Keys are always `alpha',`beta',`gamma'
            Values are length-2 tuples (mu, sigma), representing the mean and standard deviation
            of log rates.
        """ 
        super(MaskVAE, self).__init__(adata,
                                      tmax,
                                      dim_z,
                                      dim_cond,
                                      device,
                                      hidden_size,
                                      full_vb,
                                      discrete,
                                      graph_decoder,
                                      attention,
                                      init_method,
                                      init_key,
                                      tprior,
                                      init_ton_zero,
                                      filter_gene,
                                      count_distribution,
                                      std_z_prior,
                                      checkpoints,
                                      rate_prior,
                                      **kwargs)
        self.enc_mask_token = nn.Parameter(torch.empty(adata.n_vars*2, dtype=torch.float32, device=self.device))
        self.dec_mask_token = nn.Parameter(torch.empty(dim_z, dtype=torch.float32, device=self.device))
        nn.init.uniform_(self.enc_mask_token)
        nn.init.normal_(self.enc_mask_token, std=std_z_prior)
    
    def forward(self,
                data_in,
                edge_index,
                mask,
                lu_scale,
                ls_scale,
                edge_weight=None,
                u0=None,
                s0=None,
                t0=None,
                t1=None,
                condition=None):
        """Standard forward pass.

        Arguments
        ---------

        data_in : `torch.tensor`
            input count data, (N, 2G)
        mask : `torch.tensor`
            Indices of masked cells
        lu_scale : `torch.tensor`
            library size scaling factor of unspliced counts, (G)
            Effective in the discrete mode and set to 1's in the
            continuouts model
        ls_scale : `torch.tensor`
            Similar to lu_scale, but for spliced counts, (G)
        u0 : `torch.tensor`, optional
            Initial condition of u, (N, G)
            This is set to None in the first stage when cell time is
            not fixed. It will have some value in the second stage, so the users
            shouldn't worry about feeding the parameter themselves.
        s0 : `torch.tensor`, optional
            Initial condition of s, (N,G)
        t0 : `torch.tensor`, optional
            time at the initial condition, (N,1)
        t1 : `torch.tensor`, optional
            time at the future state.
            Used only when `vel_continuity_loss` is set to True
        condition : `torch.tensor`, optional
            Any additional condition to the VAE

        Returns
        -------

        mu_t : `torch.tensor`, optional
            time mean, (N,1)
        std_t : `torch.tensor`, optional
            time standard deviation, (N,1)
        mu_z : `torch.tensor`, optional
            cell state mean, (N, Cz)
        std_z : `torch.tensor`, optional
            cell state standard deviation, (N, Cz)
        t : `torch.tensor`, optional
            sampled cell time, (N,1)
        z : `torch.tensor`, optional
            sampled cell sate, (N,Cz)
        uhat : `torch.tensor`, optional
            predicted mean u values, (N,G)
        shat : `torch.tensor`, optional
            predicted mean s values, (N,G)
        """
        data_in_scale = data_in
        G = data_in_scale.shape[-1]//2
        # optional data scaling
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :G]/self.decoder.scaling.exp(),
                                       data_in_scale[:, G:]), 1)
        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :, :G]/lu_scale,
                                       data_in_scale[:, :, G:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)
        # Apply the mask
        data_in_scale[mask] = self.enc_mask_token
        data_in_scale[self.validation_idx] = self.enc_mask_token

        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale,
                                                        edge_index,
                                                        edge_weight,
                                                        condition)

        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)
        # Apply the mask
        z[mask] = self.dec_mask_token
        z[self.validation_idx] = self.dec_mask_token

        return_vel = self.config['reg_v'] or self.config['vel_continuity_loss']
        uhat, shat, vu, vs = self.decoder.forward(t[mask],
                                                  z,
                                                  mask,
                                                  edge_index,
                                                  edge_weight,
                                                  u0,
                                                  s0,
                                                  t0,
                                                  condition,
                                                  neg_slope=self.config["neg_slope"])

        if t1 is not None:  # predict the future state when we enable velocity continuity loss
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  z,
                                                                  mask,
                                                                  edge_index,
                                                                  edge_weight,
                                                                  edge_index,
                                                                  edge_weight,
                                                                  uhat,
                                                                  shat,
                                                                  t,
                                                                  condition,
                                                                  return_vel=return_vel,
                                                                  neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return (mu_t[mask], std_t[mask],
                mu_z[mask], std_z[mask],
                t[mask], z[mask],
                uhat, shat,
                uhat_fw, shat_fw,
                vu, vs,
                vu_fw, vs_fw)
    
    def train(self,
              adata,
              graph,
              config={},
              plot=False,
              gene_plot=[],
              cluster_key="clusters",
              figure_path="figures",
              embed="umap",
              random_state=2022,
              test_samples=None):
        """The high-level API for training.

        Arguments
        ---------

        adata : :class:`anndata.AnnData`
        config : dictionary, optional
            Contains all hyper-parameters.
        plot : bool, optional
            Whether to plot some sample genes during training. Used for debugging.
        gene_plot : string list, optional
            List of gene names to plot. Used only if plot==True
        cluster_key : str, optional
            Key in adata.obs storing the cell type annotation.
        figure_path : str, optional
            Path to the folder for saving plots.
        embed : str, optional
            Low dimensional embedding in adata.obsm. The actual key storing the embedding should be f'X_{embed}'
        """
        self.load_config(config)
        if self.config["learning_rate"] is None:
            p = (np.sum(adata.layers["unspliced"].A > 0)
                 + (np.sum(adata.layers["spliced"].A > 0)))/adata.n_obs/adata.n_vars/2
            self._set_lr(p)
            print(f'Learning Rate based on Data Sparsity: {self.config["learning_rate"]:.4f}')
        print("--------------------------- Train a VeloVAE ---------------------------")
        # Get data loader
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((U, S), 1).astype(int)
        else:
            X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = np.nan*np.ones((adata.n_obs, 2))
            plot = False

        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs
                           else np.array(['Unknown' for i in range(adata.n_obs)]))
        # Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])

        print("*********               Creating a Graph Dataset              *********")
        n_train = int(adata.n_obs * self.config['train_test_split'])
        self.graph_data = SCGraphData(X,
                                      self.cell_labels,
                                      graph,
                                      n_train,
                                      self.device,
                                      test_samples,
                                      self.config['train_edge_weight'],
                                      random_state)
        
        self.train_idx = self.graph_data.train_idx.cpu().numpy()
        self.validation_idx = self.graph_data.validation_idx.cpu().numpy()
        self.test_idx = test_samples
        # Automatically set test iteration if not given
        if self.config["test_iter"] is None:
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print("*********                      Finished.                      *********")

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        # define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())\
            + list(self.decoder.net_rho.parameters())\
            + [self.enc_mask_token, self.dec_mask_token]
        param_ode = [self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma,
                     self.decoder.u0,
                     self.decoder.s0,
                     self.decoder.logit_pw]
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")

        # Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        n_epochs = self.config["n_epochs"]
        start = time.time()
        for epoch in range(n_epochs):
            # Train the encoder
            if epoch >= self.config["n_warmup"]:
                stop_training = self.train_epoch(optimizer_ode, optimizer)
            else:
                stop_training = self.train_epoch(optimizer, None)

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                print('Plotting')
                elbo_train = self.test(Xembed[self.train_idx],
                                       f"train{epoch+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       plot,
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f},\t"
                      f"Test ELBO = {elbo_test:.3f},\t"
                      f"Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break

        count_epoch = epoch+1
        n_test1 = len(self.loss_test)

        print("*********                      Stage  2                       *********")
        self.encoder.eval()
        self.use_knn = True
        self.train_stage = 2
        self.decoder.logit_pw.requires_grad = False
        if not self.is_discrete:
            sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
            sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
            u0_prev, s0_prev = None, None
            noise_change = np.inf
        x0_change = np.inf
        x0_change_prev = np.inf
        param_post = list(self.decoder.net_rho2.parameters()) + [self.dec_mask_token]
        optimizer_post = torch.optim.Adam(param_post,
                                          lr=self.config["learning_rate_post"],
                                          weight_decay=self.config["lambda_rho"])
        param_ode = [self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma,
                     self.decoder.u0,
                     self.decoder.s0]
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        for r in range(self.config['n_refine']):
            print(f"*********             Velocity Refinement Round {r+1}              *********")
            self.config['early_stop_thred'] *= 0.95
            stop_training = (x0_change - x0_change_prev >= -0.01 and r > 1) or (x0_change < 0.01)
            if (not self.is_discrete) and (noise_change > 0.001) and (r < self.config['n_refine']-1):
                self.update_std_noise()
                stop_training = False
            if stop_training:
                print(f"Stage 2: Early Stop Triggered at round {r}.")
                break
            self.update_x0()
            self.n_drop = 0

            for epoch in range(self.config["n_epochs_post"]):
                if epoch >= self.config["n_warmup"]:
                    stop_training = self.train_epoch(optimizer_post, optimizer_ode)
                else:
                    stop_training = self.train_epoch(optimizer_post, None)

                if plot and (epoch == 0 or (epoch+count_epoch+1) % self.config["save_epoch"] == 0):
                    elbo_train = self.test(Xembed[self.train_idx],
                                           f"train{epoch+count_epoch+1}",
                                           False,
                                           gind,
                                           gene_plot,
                                           plot,
                                           figure_path)
                    self.decoder.train()
                    elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                    print(f"Epoch {epoch+count_epoch+1}: Train ELBO = {elbo_train:.3f},\t"
                          f"Test ELBO = {elbo_test:.3f},\t"
                          f"Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    print(f"*********       "
                          f"Round {r+1}: Early Stop Triggered at epoch {epoch+count_epoch+1}."
                          f"       *********")
                    break
            count_epoch += (epoch+1)
            if not self.is_discrete:
                sigma_u = self.decoder.sigma_u.detach().cpu().numpy()
                sigma_s = self.decoder.sigma_s.detach().cpu().numpy()
                norm_delta_sigma = np.sum((sigma_u-sigma_u_prev)**2 + (sigma_s-sigma_s_prev)**2)
                norm_sigma = np.sum(sigma_u_prev**2 + sigma_s_prev**2)
                sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
                sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
                noise_change = norm_delta_sigma/norm_sigma
                print(f"Change in noise variance: {noise_change:.4f}")
            with torch.no_grad():
                if r > 0:
                    x0_change_prev = x0_change
                    norm_delta_x0 = np.sqrt(((self.u0 - u0_prev)**2 + (self.s0 - s0_prev)**2).sum(1).mean())
                    std_x = np.sqrt((self.u0.var(0) + self.s0.var(0)).sum())
                    x0_change = norm_delta_x0/std_x
                    print(f"Change in x0: {x0_change:.4f}")
                u0_prev = self.u0
                s0_prev = self.s0

        elbo_train = self.test(Xembed[self.train_idx],
                               "final-train",
                               False,
                               gind,
                               gene_plot,
                               plot,
                               figure_path)
        elbo_test = self.test(Xembed[self.validation_idx],
                              "final-test",
                              True,
                              gind,
                              gene_plot,
                              plot,
                              figure_path)
        self.loss_train.append(-elbo_train)
        self.loss_test.append(elbo_test)
        # Plot final results
        if plot:
            plot_train_loss(self.loss_train,
                            range(1, len(self.loss_train)+1),
                            save=f'{figure_path}/train_loss_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test,
                               [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                               save=f'{figure_path}/test_loss_velovae.png')

        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        return
