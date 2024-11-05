import os
import copy
from re import L
import dill 
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from copy import deepcopy
from tqdm.auto import tqdm
from neurodiffeq import diff
import torch.nn.functional as F
from ordered_set import OrderedSet
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.generators import BaseGenerator
from neurodiffeq.callbacks import ActionCallback 
from neurodiffeq.generators import Generator1D, PredefinedGenerator
from neurodiffeq.conditions import BundleIVP, NoCondition, BundleDirichletBVP

large = 20
med = 16
small = 12

import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter

graphsize = (4, 4)
colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']
params = {'axes.titlesize': small,
          'legend.fontsize': small,
          'figure.figsize': graphsize,
          'axes.labelsize': small,
          'axes.linewidth': 2,
          'xtick.labelsize': small,
          'xtick.color' : '#1D1717',
          'ytick.color' : '#1D1717',
          'ytick.labelsize': small,
          'axes.edgecolor':'#1D1717',
          'figure.titlesize': med,
          'axes.prop_cycle': cycler(color = colors),
          'text.usetex': False,
          'font.family': 'serif',  # Choose your font family (e.g., "serif", "sans-serif", "monospace")
          'font.serif': ['Times']}# Choose your font (e.g., "Times", "Arial", "Computer Modern Roman")}

# Define your custom colormap with #66bb6a
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['#66bb6a', '#1D1717'])
plt.rcParams.update(params)

IN_COLAB = torch.cuda.is_available()

def V_or(phi, phim = None):
    if phim == None:
        phim = 1.0
    else:
        phim = phim
    phiq = 10
    term1 = 6 * (2*phi)**2 * (8 + 2*(2*phi)**2/(phim)**2 - 3*(2*phi)**4 / phiq )**2
    term2 = (96 +8*(2*phi)**2 + (2*phi)**4/phim**2 - (2*phi)**6/phiq)**2
    return (1/4)*(1/768)*(term1 - term2)

def DV_or(phi):
    phim = 1.0
    phiq = 10
    return (-(1/(3*phim**4*phiq**2))*phi*(phiq**2*phi**4*(-9 + 2*phi**2)+
            2*phim**2*phiq*phi**4*(3*phiq+72*phi**2 - 10*phi**4) +
            phim**4*(12*phi**8*(-45 + 4*phi**2) + phiq**2*(9 + 4*phi**2) -
           4*phiq*phi**4*(-9 + 8*phi**2))))

# class NetA(FCNN):
#     def __init__(self, hidden_units=[32,32,32]):
#         # Initialize as FCNN with 3 inputs (u, S, T) and 4 outputs (Σ, A, νΣ, νA)
#         super().__init__(n_input_units=3, hidden_units=hidden_units, n_output_units=4)
    
#     def forward(self, u, S, T):
#         x = torch.cat([u, S, T], dim=1)
#         return super().forward(x)

# class NetB(FCNN):
#     def __init__(self, hidden_units=[32,32,32], include_metric_vars=False):
#         # Initialize as FCNN with either 1 input (u) or 5 inputs (u, Σ, A, νΣ, νA)
#         # and 2 outputs (ϕ, νϕ)
#         n_input_units = 1 if not include_metric_vars else 5
#         super().__init__(n_input_units=n_input_units, hidden_units=hidden_units, n_output_units=2)
#         self.include_metric_vars = include_metric_vars
    
#     def forward(self, u, Sigma=None, A=None, nu_Sigma=None, nu_A=None):
#         if self.include_metric_vars:
#             x = torch.cat([u, Sigma, A, nu_Sigma, nu_A], dim=1)
#         else:
#             x = u.unsqueeze(1)
#         return super().forward(x)

class CustomNN(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))

        # Learnable parameters mu and sigma for the first layer
        #self.mu =  torch.linspace(0,1, hidden_units[0])
        self.mu = nn.Parameter(torch.linspace(0,2, hidden_units[0]))
        #self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.1)
        self.sigma = torch.ones(hidden_units[0])*0.1

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)

    def forward(self, x):

        inputx = x[:,0].reshape(-1,1)
        #print(inputx.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(x.shape)
            # Apply the custom operation after the first layer
            if i == 0:
                x = x * torch.exp(- (x - self.mu) ** 2 / self.sigma ** 2)

        # Output layer transformation
        x = self.fc_out(x)
        return x
    
class MeshGenerator(BaseGenerator):

    def __init__(self, g1, pg):

        super(MeshGenerator, self).__init__()
        self.g1 = g1
        self.pg = pg

    def get_examples(self):

        # Get u values
        u = self.g1.get_examples().reshape(-1, 1)
        batch_size = u.shape[0]

        # u = u.reshape(-1, 1, 1)

        bundle_params = self.pg.get_examples()
        if isinstance(bundle_params, torch.Tensor):
            bundle_params = (bundle_params,)

        # Stack and reshape
        bundle_params = torch.stack(bundle_params, dim=1)
        
        # Make sizes match
        if bundle_params.shape[0] != batch_size:
            bundle_params = bundle_params.expand(batch_size, -1)
        
        # Split into S and T
        S = bundle_params[:, 0].reshape(-1, 1)
        T = bundle_params[:, 1].reshape(-1, 1)

        # Verify shapes
        # assert u.shape[0] == S.shape[0] == T.shape[0], \
        #     f"Shape mismatch: u:{u.shape}, S:{S.shape}, T:{T.shape}"

        # assert len(bundle_params[0].shape) == 1, "shape error, ask shuheng"
        # n_params = len(bundle_params)

        # bundle_params = torch.stack(bundle_params, dim=1)
        # bundle_params = bundle_params.reshape(1, -1, n_params)

        # uu, bb = torch.broadcast_tensors(u, bundle_params)
        # uu = uu[:, :, 0].reshape(-1)
        # bb = [bb[:, :, i].reshape(-1) for i in range(n_params)]

        #return uu, *bb

        return u, S, T

class minmaxScaler():
  def __init__(self, x):
    self.minx = x.min().detach().item()
    self.maxx = x.max().detach().item()

  def transform(self, x):
    return (x - self.minx)/(self.maxx - self.minx)
  
class DoSchedulerStep(ActionCallback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def __call__(self, solver):
        self.scheduler.step()

class BestValidationCallback(ActionCallback):
    def __init__(self):
        super().__init__()
        self.best_potential = None

    def __call__(self, solver):
        if solver.lowest_loss is None or solver.metrics_history['r2_loss'][-1] <= solver.lowest_loss:
            self.best_potential = copy.deepcopy(solver.V)

class Store_MSE_Loss(ActionCallback):
    def __init__(self):
        super().__init__()
        self.mse_loss_history = []

    def __call__(self, solver):
        if solver.global_epoch % 10 == 0:
          for i in range(5):
            batch = self.generator['train'].get_examples()
            r = solver.get_residuals(*batch, to_numpy = True)
            self.mse_loss_history.append((np.array(r)**2).mean())

class CustomBundleSolver1D(BundleSolver1D):
    def __init__(self, *args, **kwargs):

        self.V = kwargs.pop('V', None)

        super().__init__( *args, **kwargs)
        self.metrics_history['r2_loss'] = []
        self.metrics_history['phi_max'] = []
    
    def get_solution(self, copy=True, best=True):
    # Choose which networks to use. In the original, the default method has copy, best = True but now we can actually choose
        if best:
            networks = self.best_nets  # Use best networks found during training
        else:
            networks = self.nets       # Use current networks
        
    # Get networks, optionally making copies
        if copy:
            net_a = deepcopy(networks[0])
            net_b = deepcopy(networks[1])
        else:
            net_a = networks[0]
            net_b = networks[1]

        def solution(u, Sigma_h, Va_h, to_numpy=True):
            with torch.no_grad():
                # Convert boundary values to S, T for net_a input
                S = (Sigma_h * np.pi) ** 3
                T = -Va_h / (4 * np.pi)
                
                # Get solutions from networks
                Sigma, A, nu_Sigma, nu_A = net_a(torch.cat([u, S, T], dim=1))
                phi, nu_phi = net_b(torch.cat([u, Sigma, A, nu_Sigma, nu_A], dim=1))
                
                if to_numpy:
                    return (nu_Sigma.numpy(), nu_A.numpy(), nu_phi.numpy(),
                           Sigma.numpy(), A.numpy(), phi.numpy())
                return nu_Sigma, nu_A, nu_phi, Sigma, A, phi

        return solution

    def _set_loss_fn(self, criterion):
        pass

    def loss_fn(self,r,f,x):
        
        loss_r2 = (r**2).mean() 
        self.metrics_history['r2_loss'].append(loss_r2.detach().item())

        phi = f[1][1]  # Second output from NetB is phi
        self.metrics_history['phi_max'].append(phi[-49: ].mean().detach().item())
        return loss_r2

    def _update_best(self, key):
        """Update best networks based on current loss.  
        Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history['r2_loss'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = [
                deepcopy(self.nets[0]),  # NetA
                deepcopy(self.nets[1])   # NetB
            ]

    def fit(self, max_epochs, callbacks=(), tqdm_file='default', **kwargs): # The actual training 
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training, validating and updating best model.

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]
        :param tqdm_file:
            File to write tqdm progress bar. If set to None, tqdm is not used at all.
            Defaults to ``sys.stderr``.
        :type tqdm_file: io.StringIO or _io.TextIOWrapper

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        self.callbacks = callbacks

        monitor = kwargs.pop('monitor', None)
        if monitor:
            warnings.warn("Passing `monitor` is deprecated, "
                          "use a MonitorCallback and pass a list of callbacks instead")
            callbacks = [monitor.to_callback()] + list(callbacks)
        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')  # pragma: no cover

        flag=False
        if str(tqdm_file) == 'default':
            bar = tqdm(
                total = max_epochs,
                desc='Training Progress',
                colour='blue',
                dynamic_ncols=True,
            )
        elif tqdm_file is not None:
            bar = tqdm_file
        else:
            flag=True
        
            

        for local_epoch in range(max_epochs):
             #stop training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register local epoch (starting from 1 instead of 0) so it can be accessed by callbacks
            self.local_epoch = local_epoch + 1
            self.run_train_epoch()
            self.run_valid_epoch()
            for cb in callbacks:
                cb(self)
            if not flag:
                bar.update(1)

# IMPORT DATA
df_data_yago_a1 = pd.read_csv("../Data/1st order/A_hT.txt", sep=" ", header=None).values
df_data_yago_sigma1 = pd.read_csv("../Data/1st order/Sigma_hT.txt", sep=" ", header=None).values
df_data_yago_phi1 = pd.read_csv("../Data/1st order/phi_hT.txt", sep=" ", header=None).values

A_yago1 = df_data_yago_a1[:, 1]
u_yago1 = df_data_yago_a1[:, 0]
Sigma_yago1 = df_data_yago_sigma1[:, 1]
phi_yago1 = df_data_yago_phi1[:, 1]


#point 2 (mid point)
df_data_yago_a2 = pd.read_csv("../Data/1st order/A_mT.txt", sep=" ", header=None).values
df_data_yago_sigma2 = pd.read_csv("../Data/1st order/Sigma_mT.txt", sep=" ", header=None).values
df_data_yago_phi2 = pd.read_csv("../Data/1st order/phi_mT.txt", sep=" ", header=None).values

A_yago2 = df_data_yago_a2[:, 1]
u_yago2 = df_data_yago_a2[:, 0]
Sigma_yago2 = df_data_yago_sigma2[:, 1]
phi_yago2 = df_data_yago_phi2[:, 1]

#point 3 (left point)
df_data_yago_a3 = pd.read_csv("../Data/1st order/A_lT.txt", sep=" ", header=None).values
df_data_yago_sigma3 = pd.read_csv("../Data/1st order/Sigma_lT.txt", sep=" ", header=None).values
df_data_yago_phi3 = pd.read_csv("../Data/1st order/phi_lT.txt", sep=" ", header=None).values

A_yago3 = df_data_yago_a3[:, 1]
u_yago3 = df_data_yago_a3[:, 0]
Sigma_yago3 = df_data_yago_sigma3[:, 1]
phi_yago3 = df_data_yago_phi3[:, 1]


Sigma_yago_all = [Sigma_yago1, Sigma_yago2, Sigma_yago3]
A_yago_all = [A_yago1, A_yago2, A_yago3]
phi_yago_all = [phi_yago1, phi_yago2, phi_yago3]

u_yago=u_yago1 #same as u_yago2,3

DA=[]
[DA.append(np.gradient(A_yago_all[i],u_yago)) for i in range(3)]

T_h=[]
S_h=[]
[T_h.append(DA[i][-1]/(-4*np.pi* u_yago[-1]**2)) for i in range(3)]

[S_h.append((np.pi*Sigma_yago_all[i]**3)[-1]) for i in range(3)]

#crossover
#S_yago = [8.80154741176382, 1.2506823519737085, 0.17257280631479724] 
#T_yago = [0.48423108257748665, 0.2883770837025976, 0.15629231178160138] 
#phi_uh_yago= [0.6, 1, 1.13]

#3of5
S_yago = [8.939014410418975, 1.42689,  0.31734643273483476] 
T_yago = [0.4867126785278015, 0.395869, 0.26805164866639136] 
phi_uh_yago= [0.6, 1.4114516577290581, 1.528]

Sigma_uh_yago=[]
Va_uh_yago=[]

for i in range(len(S_yago)):
    Sigma_uh_yago.append((S_yago[i]/np.pi)**(1/3))
    Va_uh_yago.append((-T_yago[i]*4*np.pi))
    

# DEFINE THE WHOLE RUTINE
class NNholo():

    def __init__(self, data_path, saving_path,sampling_method ,init_pt_curve = 55, delta = 0.0, curriculum = 1.0):
        """
        Initializes the NN model.
        - data_path: Path to input data
        - saving_path: Where to save results
        - init_pt_curve: Starting point for curve fitting 
        - delta: Parameter for boundary condition
        - curriculum: Parameter for training curriculum
        """
        self.delta = delta
        self.curriculum = curriculum
        self.path = saving_path
        self.data_path = data_path
        suffix_data_path = Path(data_path).suffix
        
        if suffix_data_path == ".csv":
            df_data = pd.read_csv(data_path, header=None).values
            print('File is .csv')
        if suffix_data_path == ".txt":
            df_data = pd.read_csv(data_path, sep=" ", header=None).values
            print('File is .txt')
            
            
        S_true_1= torch.tensor(df_data[init_pt_curve:100:1,1])
        T_true_1= torch.tensor(df_data[init_pt_curve:100:1,0])

        #S_true_2= torch.tensor(df_data[71:100:5,1])4
        #T_true_2= torch.tensor(df_data[71:100:5,0])

        S_true_3= torch.tensor(df_data[101:200:6,1])
        T_true_3= torch.tensor(df_data[101:200:6,0])

        S_true_4= torch.tensor(df_data[201::30,1])
        T_true_4= torch.tensor(df_data[201::30,0])

        # S_true_lowest = torch.tensor([0])
        # T_true_lowest = torch.tensor([0])

        self.S_true=torch.cat([S_true_1, S_true_3, S_true_4],dim=0)
        self.T_true=torch.cat([T_true_1, T_true_3, T_true_4],dim=0)
        
        self.Sigma_uh_all = (self.S_true/np.pi)**(1/3)
        self.Va_uh_all = (-self.T_true*4*np.pi)
        
        n_points = len(self.S_true)   

        self.pg = PredefinedGenerator(self.Sigma_uh_all, self.Va_uh_all) # u_0 and u_1, respectively
        self.g1 = Generator1D(n_points, 0, self.curriculum, method='chebyshev2') # This is u = [0, 1] with ORIGINALLY 48 points. More points close to 0 and 1 (both boundaries)
        self.g2 = Generator1D(n_points, 0, 1, method='equally-spaced') # ORIGINALLY 16
        self.train_generator =  MeshGenerator(self.g1, self.pg)
        self.valid_generator =  MeshGenerator(self.g2, self.pg)

        # Net A: handles metric variables (Sigma, A, nu_Sigma, nu_A)
        self.net_a = FCNN(n_input_units=3, 
                          hidden_units=[32,32,32], 
                          n_output_units=4)  # inputs: u,S,T; outputs: Sigma,A,nu_Sigma,nu_A

        # Net B: handles scalar field variables (phi, nu_phi)
        self.net_b = FCNN(n_input_units=3, # Need (u, S, T) too here! 
                          hidden_units=[32,32,32], 
                          n_output_units=2)  # inputs: u,Sigma,A,nu_Sigma,nu_A; outputs: phi,nu_phi

        self.nets = [self.net_a, self.net_b]
        
        # Defines the custom NN for the potential V. It takes 1 input (phi) and outputs 1 value (V(phi)), and has 4 hidden layers with 16 units each.  
        self.V = CustomNN(n_input_units = 1, hidden_units = [16,16,16,16] ,actv = nn.SiLU, n_output_units = 1) 
        
                # Modify conditions for new structure
        self.conditions = [
            BundleDirichletBVP(0, 1, 1, None, bundle_param_lookup=dict(u_1=0)),  # For Sigma
            BundleDirichletBVP(0, 1, 1, 0),   # For A
            BundleIVP(0, 0)    # For phi
            ]
    #     self.conditions = [
    # NoCondition(),  # no condition on Vs
    # BundleIVP(1, None, bundle_param_lookup=dict(u_0=1)), #condition on Va = -4 pi T
    # BundleIVP(0, 1),   # Vphi(0) ==1
    # BundleDirichletBVP(0, 1, 1, None, bundle_param_lookup=dict(u_1=0)),  # Sigma_{u=0} = 1, Sigma_{u=1}=(S/pi)**(1/3)
    # BundleDirichletBVP(0, 1, 1, 0),   # A (0) == 1  A(1)=0
    # BundleIVP(0, 0),  #phi(0)=0 #BundleDirichletBVP(0, 0,1, phi_yago[-1])#
    # ]       
        # Creating a list of 6 fully connected NNs for A, sigma, phi and their derivatives wrt u, respectively. 
        # Each of these networks has 3 input units and 3 hidden layers with 32 units each 
        #self.nets = [FCNN(n_input_units=3, hidden_units=[32,32,32]) for _ in range(6)] 
        
      #  self.adam = torch.optim.Adam(OrderedSet([p for net in self.nets + [self.V] for p in net.parameters()]), \
        #                lr=1e-3)#,  betas=(0.9, 0.99))
        # self.adam = torch.optim.Adam(OrderedSet([p for net in self.nets + [self.V] for p in net.parameters()]), \
        #                 lr=1e-3)#,  betas=(0.9, 0.99))
        
        # self.lbfgs = torch.optim.LBFGS(OrderedSet([p for net in self.nets + [self.V] for p in net.parameters()]), \
        #                 lr=1e-2)
        
        self.adam = torch.optim.Adam(OrderedSet([p for net in [self.net_a, self.net_b, self.V] for p in net.parameters()]), \
                    lr=1e-3)#,  betas=(0.9, 0.99))

        self.lbfgs = torch.optim.LBFGS(OrderedSet([p for net in [self.net_a, self.net_b, self.V] for p in net.parameters()]), \
                    lr=1e-2)
        
        self.solver = CustomBundleSolver1D( ode_system=self.equations,
                                            conditions=self.conditions,
                                            t_min=self.delta,
                                            t_max=1,
                                            train_generator=self.train_generator,
                                            valid_generator=self.valid_generator,
                                            optimizer=self.adam,
                                            nets=[self.net_a, self.net_b],
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V
                                        )
    def sofT_curve(self):
        """
        Plots the S vs T curve from the data and key metrics.
        Creates visualizations showing:
        - S vs T relationship
        - S/T^3 vs T 
        - Va vs Sigma plots
        """    

        print('S_min: ', min(self.S_true))
        print('Length of input s(T) curve: ',  self.S_true.shape)

        #print('(S*,T*) = ', '(',S_h,',', T_h,')')
        #[print('(S*,T*)_%i'%(i+1) ,'= ', '(',S_yago[i],',', T_yago[i],')') for i in range(len(S_yago))]
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].scatter(self.T_true.detach().numpy(), self.S_true.detach().numpy())

        soverT3_x = self.T_true
        soverT3_y = self.S_true/(self.T_true**3)

        ax[1].scatter(soverT3_x, soverT3_y)
        #plt.scatter((self.T_true),(self.S_true), color='k',s=15, label='true')
        #plt.xlabel('T')
        #plt.ylabel('s')
        plt.title(f'S_true shape: {len(self.S_true)}, Max T: {"{:.2f}".format(max(self.T_true))}')
        ax[0].set_xlabel('T')
        ax[0].set_ylabel('s')
        ax[1].set_xlabel('T')
        ax[1].set_ylabel('$s/T^3$')
        ax[0].legend()
        ax[1].legend()
        plt.show() 
        # [plt.scatter(T_yago[i], S_yago[i] ,s=15, color='r',label='(S*,T*) for  A(u), Sigma(u), phi(u) of Yago') for i in range(len(S_yago))]
        #plt.legend()
        plt.show() 

        plt.scatter((self.Va_uh_all.detach().numpy()), (self.Sigma_uh_all.detach().numpy()), color='k', s=15, label='true')
        #plt.hlines(0.78, -12.0, 0)
        plt.xlabel('$Va_h$')
        plt.ylabel('$\Sigma_h$')
        #[plt.scatter(Va_uh_yago[i], Sigma_uh_yago[i] ,s=15, color='r',label='(Sigma*,Va*) for  A(u), Sigma(u), phi(u) of Yago') for i in range(len(S_yago))]
        #plt.legend()
        plt.show()
        
        fig.savefig(f'{self.path}/sofT.png')
        
    def update_generator(self, curriculum = 1.0, valid_method = 'equally-spaced'):
        """
        Updates the data generators used for training/validation.
        - curriculum: Controls difficulty of training samples
        - valid_method: How validation points are spaced
        """

        g1 = Generator1D(128, 0, curriculum, method='chebyshev2') # HOW DOES GENERATOR1D WORK?
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
        
    def update_optimizer(self, lr = None):
        if lr == None:
            for g in self.adam.param_groups:
                print('Actual learning rate: ', g['lr'])
        else:
            for g in self.adam.param_groups:
                g['lr'] = lr
                print('Learning rate updated to: ', g['lr'])

    def set_curriculum(self, start = 0.0, end = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, start, end, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
    
    #def equations(self, Vs, Va, Vp, Sigma, A, phi, u):
    def equations(self, u, S, T):

        """
        Defines the system of differential equations to be solved.
        Takes outputs from neural networks and returns residuals of equations.
        Returns 7 equations that define the physical system.
        """
        # Reshape tensors into 2D tensors with one column and however many rows are needed. 
        # The -1 specifies an unknown dimension (i.e. take whole tensor and put into one column)

            # If called with a single *args tuple of all parameters, unpack them
        if isinstance(u, tuple):
            u, S, T = u[:3]  # Take first three arguments
    
    # Ensure inputs are properly shaped
        u = u.reshape(-1, 1) if not isinstance(u, torch.Size) else u
        S = S.reshape(-1, 1) if not isinstance(S, torch.Size) else S
        T = T.reshape(-1, 1) if not isinstance(T, torch.Size) else T
        # u = u.reshape(-1, 1)
        # S = S.reshape(-1, 1)
        # T = T.reshape(-1, 1)

        print(u, S, T)

        ORIGP_FLAG = 0

        # Get outputs from NetA
        input_tensor = torch.cat([u, S, T], dim=1)  # Shape: (batch_size, 3)
        Sigma, A, nu_Sigma, nu_A = self.net_a(input_tensor).chunk(4, dim=1) # Splits a tensor into n chunks along a specified dimension, now shape (batch_size, 1)
        
        # Get outputs from NetB using NetA's outputs
        phi, nu_phi = self.net_b(input_tensor).chunk(2, dim=1)
        
        #Sigma, A, nu_Sigma, nu_A = self.net_a(torch.cat([u, S, T], dim=1))
        #phi, nu_phi = self.net_b(torch.cat([u, Sigma, A, nu_Sigma, nu_A], dim=1))

        # Get potential from V net
        V = self.V(phi)
        # create the derivative of the V wrt to phi with the custom NN
        VF = diff(self.V(phi), phi, shape_check= False)

        # Compute the equation residuals
        eq1 = nu_Sigma - diff(Sigma, u)
        eq2 = nu_A - diff(A, u)
        eq3 = nu_phi - diff(phi, u)
        eq4 = diff(nu_Sigma, u) + (2/3) * Sigma * nu_phi**2
    
        eq5 = (u**2) * Sigma * diff(nu_A, u) + (8/3) * ((1-ORIGP_FLAG) * self.V(phi) + ORIGP_FLAG * V_or(phi)) * Sigma + \
          nu_A * (3 * u**2 * nu_Sigma - 5 * Sigma * u) + \
          A * (8 * Sigma - 6 * u * nu_Sigma)

        eq6 = u**2 * Sigma * A * diff(nu_phi, u) - Sigma * ((1-ORIGP_FLAG) * VF + ORIGP_FLAG * DV_or(phi)) + \
          nu_phi * (-3 * u * A * Sigma + u**2 * Sigma * nu_A + 3 * u**2 * nu_Sigma * A)

        eq7 = (u * nu_Sigma - Sigma) * \
          (u**2 * Sigma * nu_A + 2 * A * u**2 * nu_Sigma - 4 * u * A * Sigma) - \
          (2/3) * (u * Sigma**2) * (u**2 * A * nu_phi**2 - 2 * ((1-ORIGP_FLAG) * self.V(phi) + ORIGP_FLAG * V_or(phi)))

        return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]
    

    def get_loss(self):

        residuals = self.get_residuals()
        batch = [v.reshape(-1, 1) for v in self.valid_generator.get_examples()]
        funcs = [self.solver.compute_func_val(a, b, *batch) for a, b in zip(self.solver.nets, self.solver.conditions)]
        if IN_COLAB:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch).detach().cpu().numpy()

        else:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch).detach().numpy()
        
    def get_residuals(self, display = False):
        
        u, sigma, Va = self.valid_generator.get_examples()
        res = self.solver.get_residuals(u, sigma, Va, best=True)
        dim = int((res[0].shape[0])/16)
        res_eq = np.zeros((7, 16, dim)) 
        for i, r in enumerate(res):
            res_eq[i, :,:] =r.cpu().detach().reshape(16, dim)
        if display:
            print(f'Mean of residuals : {round((torch.cat(res) ** 2).mean().item(),9)}.')
        return res_eq
    
    def plot_residuals(self):
              
        residuals = self.get_residuals()
        
        fig, ax = plt.subplots(3,2, figsize=(6,18))
        ax = ax.flatten()

        vmax = 0.04

        levels = np.arange(0, vmax, .001)
        for eqn in np.arange(6):  
            im = ax[eqn].imshow( (np.abs(residuals[eqn,:,:].T)),  vmin=0, vmax=vmax, interpolation='bilinear', cmap=cmap)
            ax[eqn].contour(    (np.abs(residuals[eqn,:,:].T)), levels,   extend='both')
            ax[eqn].set_title(f"Eq{eqn+1}")
        # Add a colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.9, 0.2, 0.05, 0.6])  # Adjust the position of the colorbar
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    def plot_loss(self, color=None, xlabel= 'epochs', ylabel = r'$\log_{10}\mathcal{L}$', fontsize = 14, \
                  figsize=(8,6), thick=0.8, left_x_lim=0):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        trace = self.solver.metrics_history
        fig1 = plt.figure(figsize=figsize)
        if color==None:
            plt.plot(np.log10(trace['train_loss']), label='train loss')
        else:
            plt.plot(np.log10(trace['train_loss']), label='train loss', color = color)
            
        if len(trace['valid_loss'])!=0:
            plt.plot(np.log10(trace['valid_loss']), label='validation loss')
        if 'train__res_eq1' in trace:
            for i in range(7): 
                plt.plot(np.log10(trace[f'train__res_eq{i+1}']), label = f'eq{i+1} residuals', alpha=0.6)
                
        
        # Customize the x-axis ticks and labels
        #custom_ticks = np.arange(0, 3e6, 500000)  # Define custom tick locations
        #custom_labels = ['0.5', '1', '1.5', '2', '2.5', '3']  # Define custom tick labels
        #plt.gca().set_xticks(custom_ticks)  # Set custom tick locations
        #plt.gca().set_xticklabels(custom_labels)  # Set custom tick labels
        plt.gca().spines['top'].set_linewidth(thick)  # Adjust the thickness as needed
        plt.gca().spines['right'].set_linewidth(thick)
        plt.gca().spines['bottom'].set_linewidth(thick)
        plt.gca().spines['left'].set_linewidth(thick)
        plt.xlabel(xlabel, fontsize = fontsize)
        #plt.ylabel('DE Residual Square Loss')
        plt.ylabel(ylabel, fontsize = fontsize)
        #plt.xlim(left=left_x_lim,right=3e6)
        #plt.grid()
        #plt.legend(loc='upper right')
        #plt.tight_layout()
        plt.show()
        
        print('Min loss: ', min(trace['train_loss']))
        
        fig1.savefig(f'{self.path}/loss_epoch {trained_epochs}_prettier.png')
        fig1.savefig(f'{self.path}/loss_epoch {trained_epochs}_prettier.eps')
        fig1.savefig(f'{self.path}/loss_epoch {trained_epochs}_prettier.pdf')

    def plot_potential(self, phim, save_fig = True, best = False):
        """
        Plots the learned potential V(phi) against theoretical values.
        - phim: Parameter for theoretical potential
        - save_fig: Whether to save the plot
        - best: Whether to use best or current model
        """
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        u = np.linspace(0.0001, 1, 100)
        solution = self.solver.get_solution(best=True)
        phi_h = np.ones(self.S_true.shape)
        true_phi_h = np.ones(self.S_true.shape)
        u_max = np.ones(self.S_true.shape)

        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
        #    print(i,S,T)
            Sigma_v = (S/np.pi)**(1/3)
            Va_v = (-T*4*np.pi)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, to_numpy=True) # Get solutions from networks
            phi_h[i] = phi.max()
            true_phi_h[i] = phi[-1]
            i_max = phi.argmax()
            u_max[i] = u[i_max]
        print('max phi_h= ',max(phi_h))

        # Define the domain of input phi
        phi=torch.reshape(torch.linspace(0,max(phi_h)+0.3,100),[100,1])
        # phi=torch.reshape(torch.linspace(0.0,max(phi_h),100),[100,1])
        phi = torch.Tensor(phi)
        phi.requires_grad = True
        qphi = phi.cpu().detach().numpy().reshape(-1,)
        qphi.shape

        #Vv = potential_cb.best_potential(phi) #potential_cb.best_potential(phi)
        #DVv = diff(potential_cb.best_potential(phi), phi, shape_check= False)
        #DDVv = diff(potential_cb.best_potential(phi), phi, order=2, shape_check= False)
        Vv = self.V(phi) # Use V network
        #print(Vv)
        potentialVphi=pd.DataFrame(phi.cpu().detach().numpy())
        potentialVVv=pd.DataFrame(Vv.cpu().detach().numpy())
        #potentialDVVv=pd.DataFrame(DVv.cpu().detach().numpy())
        #potentialDDVVv=pd.DataFrame(DDVv.cpu().detach().numpy())

        py_listphi=phi.tolist()
        py_listV=Vv.tolist()
        #py_listDV=DVv.tolist()
        #py_listDDV=DDVv.tolist()
        #potentialV=pd.DataFrame(list(zip(py_listphi,py_listV,py_listDV,py_listDDV)),columns=['Phi','V','DV','DDV'])
        potentialV=pd.DataFrame(list(zip(py_listphi,py_listV)),columns=['Phi','V'])
        potentialV['Phi']=potentialV['Phi'].str[0]
        potentialV['V']=potentialV['V'].str[0]
        #potentialV['DV']=potentialV['DV'].str[0]
        #potentialV['DDV']=potentialV['DDV'].str[0]
        potentialV.to_csv(f'{self.path}/V_epoch {trained_epochs}.csv', index=False)
        
        fig2=plt.figure(figsize = (6,4))
        plt.plot(qphi,  V_or(qphi, phim).reshape(-1,1), color = 'blue', label = 'Theory')
        plt.plot(qphi, Vv.cpu().detach(), label = 'NN', color ='orange')
        #plt.plot(phi.detach(),   (1- ORIGP_FLAG)* 1*Vv.detach() +  ORIGP_FLAG *  V_or(qphi).reshape(-1,1), label='Our Solution',color='black');
        #plt.plot(qphi, 1.*V_or(qphi), label='Known')
        #plt.plot(phi.detach()[70:99], -1.0*np.exp(1.68*phi.detach()[70:99]),label='$-\exp(1.68 \phi$)',color='blue')
        #plt.vlines(1.4,-20,-1)
        plt.xlabel('$\phi$')
        plt.ylabel('$V(\phi)$')
        
        phi_h_th = qphi[np.where(V_or(qphi,phim)==min(V_or(qphi,phim)))]
        plt.axvline(phi_h_th, color = 'blue', linestyle = 'dashed', label='$\phi_{h,th}=%.2f$'%phi_h_th)
        plt.axvline(max(phi_h), label='$\phi_{h,NN}=$%.2f'%max(phi_h), color='orange', linestyle = 'dashed')
        plt.axhline(min(V_or(qphi, phim)), color ='blue', linestyle ='dotted', label='$V_{th}^{min}=%.2f$'%min(V_or(qphi,phim)))
        
        #print('len V_NN', len(Vv),type(Vv))
        #print('len phi: ', len(phi), type(phi))
        #print('len V_NN', len(Vv.cpu().detach().numpy()),type(Vv.cpu().detach().numpy()))
        #print('len phi: ', len(phi.detach().numpy()), type(phi.detach().numpy()))
        #print(phi.detach().numpy()[:,0])
        #print('len phi_h: ', len(phi_h))
        
        V_nn_interp = interp1d(phi.detach().numpy()[:,0], Vv.cpu().detach().numpy()[:,0])
        V_nn_h = V_nn_interp(phi_h)
        plt.axhline(V_nn_h[int(np.where(phi_h==max(phi_h))[0])], color = 'orange', linestyle ='dotted', label='$V_{NN}^{min}=%.2f$'%V_nn_h[int(np.where(phi_h==max(phi_h))[0])])

        plt.ylim((min(V_or(qphi,phim))-1, -2))
        plt.xlim((0.0,max(phi_h)+0.1))
        plt.legend()
        
        if save_fig == True:
            fig2.savefig(f'{self.path}/V_epoch {trained_epochs}.png')
        
        
    def plot_colored_sofT(self, phiM_chosen, colormap = 'vidris', fontsize = 14, n_fontsize = 14, \
                          dot_size=10, thick=0.8, figsize = (8,6)):
        
        u = np.linspace(0.0001, 1, 100)
        phi_h = np.ones(self.S_true.shape)
        solution = self.solver.get_solution(best=True)
        
        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
            Sigma_v = (S/np.pi)**(1/3)
            Va_v = (-T*4*np.pi)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, to_numpy=True)
            phi_h[i] = phi.max()
            #true_phi_h[i] = phi[-1]
            #i_max = phi.argmax()
            #u_max[i] = u[i_max]

        #plt.scatter(phi_h, self.T_true.cpu(), color=color)
        #plt.xlabel(r'$\phi_H$')
        #plt.ylabel('T')
        #plt.show()

        #plt.scatter(phi_h, self.S_true.cpu())
        #plt.xlabel('phih')
        #plt.ylabel('S')
        #plt.show()

        step = 1
        fig = plt.figure(figsize=figsize)
        plt.scatter((self.T_true.cpu()[::step]),(self.S_true.cpu()[::step]), c=phi_h[::step], s=dot_size, cmap=colormap)
        #plt.hlines(1.5028, min(T_true), max(T_true))
        colorbar = plt.colorbar()
        colorbar.set_label(r'$\phi_H$', fontsize = fontsize)
        colorbar.ax.yaxis.label.set_rotation(0) 
        plt.xlabel(r'$T/\Lambda$', fontsize=fontsize)
        plt.ylabel(r'$S/\Lambda^3$', fontsize=fontsize)
        plt.xticks(fontsize=n_fontsize)  # Adjust the font size as needed
        plt.yticks(fontsize=n_fontsize)
        plt.gca().spines['top'].set_linewidth(thick)  # Adjust the thickness as needed
        plt.gca().spines['right'].set_linewidth(thick)
        plt.gca().spines['bottom'].set_linewidth(thick)
        plt.gca().spines['left'].set_linewidth(thick)
        #plt.legend()
        fig.savefig(f'{self.path}/colored s(T) by phi_h (phiM={phiM_chosen}).pdf')

        plt.show()
        
    def plot_s_over_phi_h(self, phiM_chosen, color = 'k', fontsize = 14, n_fontsize = 14, \
                          dot_size=10, thick=0.8, figsize = (8,6)):

        u = np.linspace(0.0001, 1, 100)
        phi_h = np.ones(self.S_true.shape)
        solution = self.solver.get_solution(best=True)

        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
            Sigma_v = (S/np.pi)**(1/3)
            Va_v = (-T*4*np.pi)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, to_numpy=True)
            phi_h[i] = phi.max()
            #true_phi_h[i] = phi[-1]
            #i_max = phi.argmax()
            #u_max[i] = u[i_max]

        #plt.scatter(phi_h, self.T_true.cpu())
        #plt.xlabel('phih')
        #plt.ylabel('T')
        #plt.show()
        fig = plt.figure(figsize=figsize)
        plt.scatter(phi_h, self.S_true.cpu(), color = color, s=dot_size)
        plt.xlabel(r'$\phi_H$', fontsize= fontsize)
        plt.ylabel(r'$S/\Lambda^3$', fontsize= fontsize)
        #plt.legend(fontsize = 10)
        #plt.xlim(left=min(phi_h),right=max(phi_h))
        #plt.ylim(bottom=min(self.S_true.cpu()),top=max(self.S_true.cpu()))

        plt.xticks(fontsize=n_fontsize)  # Adjust the font size as needed
        plt.yticks(fontsize=n_fontsize)
        plt.gca().spines['top'].set_linewidth(thick)  # Adjust the thickness as needed
        plt.gca().spines['right'].set_linewidth(thick)
        plt.gca().spines['bottom'].set_linewidth(thick)
        plt.gca().spines['left'].set_linewidth(thick)
        plt.show()

        fig.savefig(f'{self.path}/s(phi_h) (phiM={phiM_chosen}).pdf')

        
    def compare_to_yago(self, fontsize = 14, legend_fontsize=14, n_fontsize=14, wspace=0.5, yago_linewidth = 3, yago_style = '--'):
        solution = self.solver.get_solution(best=True)
        
        for i in range(3):
            
            fig, ax = plt.subplots(1,2, figsize=(16,7))
            pt=i+1
            #u = np.linspace(0.0001, 1, len(u_yago))
            u = u_yago
            #S_yago_sol = 1.42689 
            #T_yago_sol = 0.395869
            #phi_uh_yago=1.4114516577290581

            #print('here', phi_uh_yago)

            S_tt=S_yago[pt-1]
            T_tt=T_yago[pt-1]

            Sigma_h = (S_tt*np.ones_like(u)/np.pi)**(1/3)
            Va_h = (-T_tt*np.ones_like(u)*4*np.pi)

            #Sigma_h = .78*np.ones_like(u)
            #Va_h = -10.0*np.ones_like(u)

            Sigma, A, phi, nu_Sigma, nu_A, nu_phi = solution(u, Sigma_h, Va_h)

            #Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_h,  Va_h, to_numpy=True)

            print('Point %i' %pt, '; phi_h_yago = %f' %phi_uh_yago[pt-1])

            ax[0].plot(u, Sigma, 'r-', label=r'$\tilde{\Sigma}_{NN}$', zorder=2)
            ax[0].plot(u_yago , Sigma_yago_all[pt-1], 'r', linestyle = yago_style, label=r'$\tilde{\Sigma}_{th}$', linewidth = yago_linewidth, zorder=1)

            ax[0].plot(u, A, 'b-', label=r'$\tilde{A}_{NN}$', zorder=2)
            ax[0].plot(u_yago, A_yago_all[pt-1], 'b', linestyle = yago_style, label = r'$\tilde{A}_{th}$', linewidth = yago_linewidth, zorder=1) 

            ax[0].plot(u, phi, 'g-', label=r'$\phi_{NN}$', zorder=2)
            ax[0].plot(u_yago, phi_yago_all[pt-1], 'g', linestyle = yago_style, label=r'$\phi_{th}$', linewidth = yago_linewidth, zorder=1)
            #ax[0].ticklabel_format(axis='y', style='sci', scilimits=(1,1))
           
            ax[0].set_xlabel('u', fontsize = fontsize)
            ax[0].set_ylabel('ODEs solutions', fontsize = fontsize)
            ax[0].set_xlim(-0.05,1.05)
            ax[0].set_ylim(-0.05,max(max(Sigma),max(A),max(phi))+0.05)
            
            #ax[0].set_title(title,fontsize = fontsize)
            ax[0].legend(fontsize = legend_fontsize)
            #plt.grid()
            #plt.savefig('solution3.png')

            #ax[1].title('Squared residuals')
            MSE_Sigma = (1/len(Sigma))*sum((Sigma_yago_all[pt-1]-Sigma)**2)
            MSE_A = (1/len(A))*sum((A_yago_all[pt-1]-A)**2)
            MSE_phi = (1/len(phi))*sum((phi_yago_all[pt-1]-phi)**2)
            
            rel_err_Sigma = (abs(Sigma_yago_all[pt-1]-Sigma)/Sigma_yago_all[pt-1])
            rel_err_A = (abs(A_yago_all[pt-1]-A)/A_yago_all[pt-1])
            rel_err_phi = (abs(phi_yago_all[pt-1]-phi)/phi_yago_all[pt-1])
            
            ax[1].plot(u,(Sigma_yago_all[pt-1]-Sigma)**2, color= 'r', label=r'$MSE_{\tilde{\Sigma}}=%.1e$'%MSE_Sigma)
            ax[1].plot(u,(A_yago_all[pt-1]-A)**2, color= 'b', label=r'$MSE_{\tilde{A}}=%.1e$'%MSE_A)
            ax[1].plot(u,(phi_yago_all[pt-1]-phi)**2, color= 'g', label=r'$MSE_{\phi}=%.1e$'%MSE_phi)
            #ax[1].ticklabel_format(axis='y', style='sci', scilimits=(1,1))
            ax[1].set_xlabel('u',fontsize = fontsize)
            ax[1].set_ylabel(r'(theory$-$NN)$^2$',fontsize = fontsize)
            ax[1].set_xlim(0,1)
            ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax[1].yaxis.get_offset_text().set_fontsize(n_fontsize)

            ax[1].legend(fontsize = legend_fontsize)
            
            title = r"$(T_{:.0f}, S_{:.0f})=({:.2f}, {:.2f})$".format(i+1,i+1,T_tt, S_tt)
            plt.suptitle(title, fontsize = fontsize)
            
            for ax in ax:
                ax.tick_params(axis='both', which='major', labelsize=n_fontsize)  # Adjust the font size
            plt.subplots_adjust(wspace=wspace)  # Increase or decrease the value to adjust the spacing
            plt.show()
            
            fig.savefig(f'{self.path}/DE solution_pt_{i+1}_(best).pdf')
            
    def plot_V_theory(self, phim = 1):
        
        col=['k','b','g','m','r','y']
        
        phim_param = phim
        #labels=['1st order s(T)','Crossover s(T)']
        plt.figure()
        phi_mins=[]
        for i in range(len(phim_param)):
            phim=phim_param[i]
            phi_th=np.linspace(0,3,250)
            V_th=V_or(phi_th, phim)
            V_phi_min=V_th[0]
            cnt=0
            for j in range(len(phi_th)):
                if V_th[j]<V_phi_min:
                    V_phi_min=V_th[j]
                    phi_min=phi_th[j]
                    cnt=j
            phi_mins.append(phi_min)
            plt.plot(phi_th[0:cnt+1],V_th[0:cnt+1],color=col[i],label='$\phi_M$=%.2f'%phim)
            plt.axvline(phi_min, linestyle = 'dotted', color=col[i])
            #plt.ylim(-8,-2)
            plt.xlim(0,max(phi_mins)+0.1)
            plt.grid()
            plt.legend()
        plt.show()
        
    def plot_residuals_in_u(self, max_bound = 0.05, print_overbound = False, save_plot=False):
       ### RESIDUAL PLOTS
        # PICK ANY
        u = np.linspace(0.0001, 1, 250)
        
        if max_bound != False:
            bottom, top = -max_bound, max_bound
        
        fig = plt.figure()
        for q,i in enumerate(zip(self.Sigma_uh_all, self.Va_uh_all)):
            Sigma_h = i[0].detach()*np.ones_like(u)
            Va_h = i[1].detach()*np.ones_like(u)

            if q % 5 == 0:
                #print(q)
                res1 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[0].detach().numpy()
                res2 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[1].detach().numpy()
                res3 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[2].detach().numpy()
                res = [res1,res2,res3]

                plt.plot(u, res1 , 'r-' ,  alpha=0.1,label='Vs Eq1')
                plt.plot(u, res2 , 'b-',  alpha=0.1, label='Va Eq2')
                plt.plot(u, res3 ,'g-', alpha=0.1,label='Vp Eq3')
                
                if q==0:
                    plt.legend()
                
                if print_overbound == True:
                    for k in range(len(res)):
                        for j in range(len(res[k])):
                            if abs(res[k][j]) > top:
                                print('For u =', u[j], ', residual(eq.',k,')=',abs(res[k][j]))
                            
            if max_bound != False:    
                plt.ylim(bottom,top)
            plt.xlim(0,1)
            plt.xlabel('u')
            plt.ylabel('DE residual')
                
        plt.show()
        
        if save_plot==True:
            fig.savefig('DE residuals 1-3.pdf')
        
        fig = plt.figure()
        for q,i in enumerate(zip(self.Sigma_uh_all, self.Va_uh_all)):

            Sigma_h = i[0].detach()*np.ones_like(u)
            Va_h = i[1].detach()*np.ones_like(u)
                
            if q % 5 ==0:
                #print(q)
                
                res4 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[3].detach().numpy()
                res5 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[4].detach().numpy()
                res6 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[5].detach().numpy()
                res7 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[6].detach().numpy()
                res = [res4,res5,res6, res7]
                   
                plt.plot(u, res4  , 'b-', alpha=0.1 ,label='Eq4')
                plt.plot(u, res5 ,   'r-', alpha=0.1, label='Eq5' )
                plt.plot(u, res6  ,  'g-', alpha=0.1, label='Eq6' )
                plt.plot(u, res7  ,  'k-', alpha=0.1, label='Eq7' )
                
                if q==0:
                    plt.legend()
                
                if print_overbound == True:
                    for k in range(len(res)):
                        for j in range(len(res[k])):
                            if abs(res[k][j]) > top:
                                print('For u =', u[j], ', residual(eq.',k,')=',abs(res[k][j]))
                            
            if max_bound != False:    
                plt.ylim(bottom,top)
            plt.xlim(0,1)
            plt.xlabel('u')
            plt.ylabel('DE residual')
            
            if save_plot==True:
                fig.savefig('DE residuals 4-7.pdf')
        
    def render(self):

        self.plot_loss()
        self.plot_residuals()
        self.plot_result()
        self.compare_to_yago()

    def save_results(self, path):

        self.solver.save(path=path)
        with open(path, 'rb') as file:
            data = dill.load(file)
        os.remove(path)
        data['net_a'] = self.net_a.state_dict()
        data['net_b'] = self.net_b.state_dict()
        data['V'] = self.V.state_dict()
        # try:
        #     data['V_best'] = self.solver.callbacks[0].best_potential.state_dict()
        #     data['V_latest'] = self.V.state_dict()

        # except:
        #     data['V_latest'] = self.V.state_dict()
        with open(path, 'wb') as file:
            dill.dump(data, file)

    def load_results(self, path):

        with open(path, 'rb') as file:
            data = dill.load(file)

        self.saved_data = data   
        
        try:     
            self.V.load_state_dict(data['V_best'])
        except:
            self.V.load_state_dict(data['V_latest'])

        train_generator = data['generator']['train']
        valid_generator = data['generator']['valid']
        de_system = data['diff_eqs']
        cond = data['conditions']
        nets = data['nets']
        best_nets = data['best_nets']
        train_loss = data['train_loss_history']
        valid_loss = data['valid_loss_history']
        optimizer = data['optimizer_class'](OrderedSet([p for net in data['nets'] + [self.V] for p in net.parameters()]))
        optimizer.load_state_dict(data['optimizer_state'])
        if data['generator']['train'].generator:
            t_min = data['generator']['train'].generator.__dict__['g1'].__dict__['t_min']
            t_max = data['generator']['train'].generator.__dict__['g1'].__dict__['t_max']
        else:
            t_min = data['generator']['train'].__dict__['g1'].__dict__['t_min']
            t_max = data['generator']['train'].__dict__['g1'].__dict__['t_max']

        self.solver = CustomBundleSolver1D( ode_system=de_system,
                                            conditions=cond,
                                            t_min=t_min,
                                            t_max=t_max,
                                            train_generator=train_generator,
                                            valid_generator=valid_generator,
                                            optimizer=optimizer,
                                            nets=[self.net_a, self.net_b],
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V
                                        )

        if best_nets != None:
            self.solver.best_nets = best_nets
        self.solver.metrics_history['train_loss'] = train_loss
        self.solver.metrics_history['valid_loss'] = valid_loss
        self.solver.diff_eqs_source = data['diff_equation_details']['equation']
