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
#from neurodiffeq import diff
import torch.nn.functional as F
from ordered_set import OrderedSet
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import BundleSolver1D
from generators import BaseGenerator,Generator1D, PredefinedGenerator
from neurodiffeq.callbacks import ActionCallback 
from neurodiffeq.conditions import BundleIVP, NoCondition, BundleDirichletBVP
import itertools
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad

large = 20
med = 16
small = 12

import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
from neurodiffeq import diff
import random

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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

def DV_or(phi, phim=1.0):
    phiq = 10
    return (-(1/(3*phim**4*phiq**2))*phi*(phiq**2*phi**4*(-9 + 2*phi**2)+
            2*phim**2*phiq*phi**4*(3*phiq+72*phi**2 - 10*phi**4) +
            phim**4*(12*phi**8*(-45 + 4*phi**2) + phiq**2*(9 + 4*phi**2) -
           4*phiq*phi**4*(-9 + 8*phi**2))))


######## WRONG !!! ########
# def DDV_or(phi,phim=1.0):
#     phiq = 10
#     p=phi
#     return -3 - p**2 - (11 * p**10) / (64 * phiq**2) + \
#             (15 * p**8 * (27 * phim**2 + phiq)) / (64 * phim**2 * phiq**2) - \
#             (7 * p**6 * (72 * phim**2 - 16 * phim**4 + phiq)) / (96 * phim**4 * phiq) - \
#             (5 * p**4 * (12 * phim**4 - 3 * phiq + 2 * phim**2 * phiq)) / (16 * phim**4 * phiq)


def DDV_or(phi, phim=None):
    if phim is None:
        phim = 1.0
    phiq = 10
    
    # Pre-calculating recurring sub-expressions for clarity
    expr_a = 8 + (8 * phi**2) / phim**2 - (48 * phi**4) / phiq
    expr_b = (16 * phi) / phim**2 - (192 * phi**3) / phiq
    expr_c = 64 * phi + (64 * phi**3) / phim**2 - (384 * phi**5) / phiq
    expr_d = 64 + (192 * phi**2) / phim**2 - (1920 * phi**4) / phiq
    expr_e = 96 + 32 * phi**2 + (16 * phi**4) / phim**2 - (64 * phi**6) / phiq
    expr_f = 16 / phim**2 - (576 * phi**2) / phiq

    # Building the main terms
    term1 = 192 * phi * expr_b * expr_a
    term2 = 48 * (expr_a)**2
    term3 = 2 * (expr_c)**2
    term4 = 2 * expr_d * expr_e
    term5 = 24 * phi**2 * (2 * (expr_b)**2 + 2 * expr_f * expr_a)

    return (1/3072) * (term1 + term2 - term3 - term4 + term5)


class Super_net(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units,diss_net):
        super(Super_net, self).__init__()
        self.core_net = FCNN(n_input_units=n_input_units, hidden_units=hidden_units,n_output_units = n_output_units,
                             actv = actv)
        self.Diss_net = diss_net
        

    def forward(self, x):
        #ind = torch.linspace(0,len(x[:,1])-1,len(x[:,1]-1)).reshape(-1,1)
        #if not(x[0,1] == x[1,1]):
        #    x[:,1] = (torch.max(x[:,1]+1e-15).detach().item()-x[:,1])/(torch.max(x[:,1]).detach().item()-torch.min(x[:,1]).detach().item())
        #    x[:,2] = (torch.max(x[:,2]+1e-15).detach().item()-x[:,2])/(torch.max(x[:,2]).detach().item()-torch.min(x[:,2]).detach().item())
        #x = torch.cat((x[:,0].reshape(-1,1),(x[:,1]/x[:,2]).reshape(-1,1),(x[:,2]).reshape(-1,1)),dim = 1)
        #print(x)
        #diss_out = self.Diss_net(diss_input)
        #diss_out = (-torch.min(diss_out)+diss_out)/(torch.max(diss_out)-torch.min(diss_out))
        #net_in = torch.cat((x[:,0].reshape(-1,1),diss_out[:,0].reshape(-1,1),diss_out[:,1].reshape(-1,1)),dim =1)
        x = self.core_net(x)
        return x

class CustomNN(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()
        self.hidden_units = hidden_units

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))
        self.hidden_units = hidden_units
        # Learnable parameters mu and sigma for the firs layer
        self.mu = nn.Parameter(torch.linspace(0,2.0, self.hidden_units[0]))
        #self.mu =  torch.linspace(0,1, hidden_units[0])
        #self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.1)
        self.sigma = torch.ones(hidden_units[0])*(2/self.hidden_units[0])

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)
        self.count = 1

    def forward(self, x):

            inputx = x[:,0].reshape(-1,1)

            for i, layer in enumerate(self.layers):
                x = layer(x)
                #print(x.shape)
                # Apply the custom operation after the first layer
                if i == 0:
                    #w = layer.weight.reshape(-1,)
                    #norm_factor = (1/torch.sqrt(2*torch.pi * (self.sigma.detach()) ** 2))
                    x = x * torch.exp(- ((x - self.mu) ** 2) / (2 * (self.sigma**2))) 

            # Output layer transformation
            x = self.fc_out(x)
            self.count +=1
            return x
    

def V1_compute_derivatives(V1, phi_c, order):

    phi_mock = torch.linspace(0, phi_c.item(), 1000, requires_grad=True).reshape(-1,1)
    V1c = V1(phi_mock)
    Vc_der_list = [V1c[-1].cpu().detach()]

    for i in range(1, order + 1):
        V1_der = diff(V1c, phi_mock, order=i)
        Vc_der_list.append(V1_der[-1].cpu().detach())

    # Turn into tensor [order] shape
    coeffs = torch.stack(Vc_der_list)  # shape (order,)

    return coeffs

def V_matching_derivatives(V2, phi, phi_c, derivatives):

    order = derivatives.shape[0]

    derivatives = derivatives.squeeze(-1)  # shape (order,)

    # print('derivatives.shape', derivatives.shape)

    pc = torch.ones_like(phi) * phi_c

    # Powers (phi - phi_c)^k / k!
    dx = (phi - pc)
    ks = torch.arange(order) #, dtype=phi.dtype, device=phi.device)
    factorials = torch.lgamma(ks + 1).exp()  # vector of k!
    terms = (dx.unsqueeze(-1) ** ks) / factorials  # shape (N, order)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    terms = terms.to(device)
    derivatives = derivatives.to(device)


    # Sum over all terms except last
    expansion = (terms * derivatives).sum(dim=-1)

    # print('expansion.shape', expansion.shape)
    # print('V2.shape', V2.shape)

    # Add last term with V2(phi)
    V2_renorm = expansion + (dx**order / torch.lgamma(torch.tensor(order+1.0)).exp()) * V2

    # print('V2_renorm.shape', V2_renorm.shape)
    
    return V2_renorm


class Second_Branch_V(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units, First_branch_V_NN, phi_c, order=3):
        super(Second_Branch_V, self).__init__()

        self.phi_c = phi_c
        self.First_branch_V_NN = First_branch_V_NN

        # phi_mock = torch.linspace(0,self.phi_c.item(), 1000, requires_grad=True).reshape(-1,1)
        # V1c = self.First_branch_V_NN(phi_mock)
        # self.Vc = V1c[-1].detach()
        # self.DVc = diff(V1c, phi_mock, order=1)[-1].detach()
        # self.DDVc = diff(diff(V1c, phi_mock, order=1), phi_mock, order=1)[-1].detach()

        self.derivatives = V1_compute_derivatives(V1 = self.First_branch_V_NN, phi_c = self.phi_c, order = order)
        
        for param in self.First_branch_V_NN.parameters():
            param.requires_grad = False

        # Layers list to hold all layers
        self.layers = nn.ModuleList()
        self.hidden_units = hidden_units

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))
        self.hidden_units = hidden_units
        # Learnable parameters mu and sigma for the firs layer
        self.mu = nn.Parameter(torch.linspace(0,3.0, self.hidden_units[0]))
        #self.mu =  torch.linspace(0,1, hidden_units[0])
        #self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.1)
        self.sigma = torch.ones(hidden_units[0])*(2/self.hidden_units[0])

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)
        self.flag = True

    def forward(self, x):

        inputx = x[:,0].reshape(-1,1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(x.shape)
            # Apply the custom operation after the first layer
            if i == 0:
                #w = layer.weight.reshape(-1,)
                #norm_factor = (1/torch.sqrt(2*torch.pi * (self.sigma.detach()) ** 2))
                x = x * torch.exp(- ((x - self.mu) ** 2) / (2 * (self.sigma**2))) 

        # Output layer transformation
        V2 = self.fc_out(x)

        # V2_renorm = self.Vc * torch.ones_like(inputx) + self.DVc * (inputx - self.phi_c*torch.ones_like(inputx)) \
        #             + (1/2) * self.DDVc * (inputx - self.phi_c*torch.ones_like(inputx))**2 + \
        #             + (1/6) * (inputx - self.phi_c*torch.ones_like(inputx))**3 * V2
        
        V2_renorm = V_matching_derivatives(V2 = V2, phi = inputx, phi_c=self.phi_c, derivatives = self.derivatives)

        a = 1000
        sigmoid1 = torch.sigmoid( - a*(inputx - self.phi_c))
        sigmoid2 = torch.sigmoid( a*(inputx - self.phi_c))

        V1 = self.First_branch_V_NN(inputx)
        
        Vtot = V1 * sigmoid1 + V2_renorm * sigmoid2
        
        return Vtot



class CustomNets(nn.Module):
    def __init__(self, n_input_units=4, hidden_units=[64,64,64], actv=nn.Tanh, n_output_units=1, loc_var = 3):
        super(CustomNets, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()
        self.n_input_units = n_input_units
        self.loc_var = loc_var

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))
        self.hidden_units = hidden_units


        # # Learnable parameters mu and sigma for the first layer
        #self.mu = nn.Parameter(torch.linspace(0.0,1.0, self.hidden_units[0]))
        
        # FIXED parameters mu and sigma for the first layer
        self.mu = (torch.linspace(0.0, 1.0, self.hidden_units[0])) #.expand(-1, n_input_units) 
        self.sigma = torch.ones(self.hidden_units[0]) * (1/self.hidden_units[0])

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)

    def forward(self, x):

        # Options for Localization variables are (u-> 0, S-> 1, T-> 2, Z->3 )
        inputx = x[:,self.loc_var].reshape(-1,1)
        #print('inputx', inputx.shape)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                w = layer.weight[:,self.loc_var]
                # print('w b4 reshape', w.shape)
                # print('w', w.shape)
                #print('mu', self.mu.shape)

                #norm_factor = (1/torch.sqrt(2*torch.pi * (self.sigma.detach()) ** 2))
                x = x * torch.exp(- (w**2)*((inputx - self.mu)** 2) / (2 * (self.sigma**2))) 

        # Output layer transformation
        x = self.fc_out(x)
        return x


class TrackNetParameters(ActionCallback):
    def __init__(self):
        super().__init__()
        self.mu_history = []
        self.sigma_history = []
        self.w_history = []
        self.grad_history = []
        
    def __call__(self, solver):
        # For mu
        for i in range(6):
            self.mu_history.append(copy.deepcopy(solver.nets[i].mu.detach().cpu().numpy()))

            # For weights (w)
            first_layer = solver.nets[i].NN[0]
            grads_hist = []
            self.w_history.append(copy.deepcopy(first_layer.weight.detach().cpu().numpy()))
    
            for name, param in solver.nets[i].named_parameters():
                if param.grad is not None:
                    # grads_net.append(copy.deepcopy(param.grad.detach().abs().mean().item()))  # mean absolute gradient
                    val=copy.deepcopy(param.grad.detach().abs().reshape(-1,1).detach().numpy())
                    grads_hist.append(val)
    
            self.grad_history.append(np.concatenate(grads_hist))

class TrackParameters(ActionCallback):
    def __init__(self):
        super().__init__()
        self.mu_history = []
        self.sigma_history = []
        self.w_history = []
        self.grad_history = []
        
    def __call__(self, solver):
        # For mu
        self.mu_history.append(copy.deepcopy(solver.V.mu.detach().cpu().numpy()))
        
        # For sigma
        self.sigma_history.append(copy.deepcopy(solver.V.sigma.detach().cpu().numpy()))

        # For weights (w)
        first_layer = solver.V.layers[0]
        grads_hist = []
        self.w_history.append(copy.deepcopy(first_layer.weight.detach().cpu().numpy()))

        for name, param in solver.V.named_parameters():
            if param.grad is not None:
                # grads_net.append(copy.deepcopy(param.grad.detach().abs().mean().item()))  # mean absolute gradient
                val=copy.deepcopy(param.grad.detach().abs().reshape(-1,1).detach().numpy())
                grads_hist.append(val)

        self.grad_history.append(np.concatenate(grads_hist))



class MeshGenerator(BaseGenerator):

    def __init__(self, g1, pg):

        super(MeshGenerator, self).__init__()
        self.g1 = g1
        self.pg = pg

    def get_examples(self):

        u = self.g1.get_examples()
        u = u.reshape(-1, 1, 1)

        bundle_params = self.pg.get_examples()
        if isinstance(bundle_params, torch.Tensor):
            bundle_params = (bundle_params,)
        assert len(bundle_params[0].shape) == 1, "shape error, ask shuheng"
        n_params = len(bundle_params)

        bundle_params = torch.stack(bundle_params, dim=1)
        bundle_params = bundle_params.reshape(1, -1, n_params)

        uu, bb = torch.broadcast_tensors(u, bundle_params)
        uu = uu[:, :, 0].reshape(-1)
        bb = [bb[:, :, i].reshape(-1) for i in range(n_params)]

        return uu, *bb

class minmaxScaler():
  def __init__(self, x):
    self.minx = x.min().detach().item()
    self.maxx = x.max().detach().item()
    self.x = x

  def transform(self):
    return (self.x - self.minx)/(self.maxx+1e-200 - self.minx)
  
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
    def __init__(self, contrastive_weights=None, u_pts = 48, mono_phi_coef=1e-5, FB_coef = 0.1, \
                  above_phi_c_coef = 0.1, indx_init_SB = 0, force_V_min_coef=0, force_DV_min_coef=0, force_DDV_min_coef=0, force_V0_coef=0, order_phi_H_coef = 0, \
                  steep_step=100, staging_addloss_epoch = 100_000, phim = None, *args, **kwargs):

        self.V = kwargs.pop('V', None)
        
        self.Vth_coef = 0

        self.mono_coef = mono_phi_coef

        self.steep_step = steep_step
        if self.mono_coef != 0:
            print(f'Monotonic phi (in S(T) direction) ADD LOSS Active! (with steepness coef:{self.steep_step})')
            
        self.FB_coef = FB_coef
        if self.FB_coef != 0:
            print('First Branch ADD LOSS Active!')

        self.above_phi_c_coef = above_phi_c_coef
        if self.above_phi_c_coef != 0:
            print('Above phi_c ADD LOSS Active!')
        self.indx_init_SB = indx_init_SB

        self.force_V_min_coef = force_V_min_coef
        if self.force_V_min_coef != 0:
            print('V_min addloss Active!')

        self.force_DV_min_coef = force_DV_min_coef
        self.force_DDV_min_coef = force_DDV_min_coef

        if self.force_DV_min_coef != 0:
            print('DV_min addloss Active!')
        if self.force_DDV_min_coef != 0:
            print('DDV_min addloss Active!')

        self.force_V0_coef = force_V0_coef
        if self.force_V0_coef != 0:
            print('V(0) forzing addloss Active!')

    
        self.order_phi_H_coef = order_phi_H_coef
        if self.order_phi_H_coef != 0:
            print('Ordering monotonically phiH Addloss Active!')


        self.staging_addloss_epoch = staging_addloss_epoch
        self.x0 = torch.tensor([self.staging_addloss_epoch])   # Epoch at which loss starts gaining weight

        
        
        self.count = 0
        super().__init__( *args, **kwargs)
        self.metrics_history['r2_loss'] = []
        self.metrics_history['phi_max'] = []
        self.metrics_history['add_loss'] = []
        self.metrics_history['monotonic_phi_add_loss'] = []
        self.metrics_history['FB_addloss'] = []
        self.metrics_history['above_phi_c_addloss'] = []
        self.metrics_history['V_min_addloss'] = []
        self.metrics_history['V0_addloss'] = []
        self.metrics_history['phiH_ordering_addloss'] = []
        # self.metrics_history['DV_DDV_min_addloss'] = []

        self.metrics_history['DV_min_addloss'] = []
        self.metrics_history['DDV_min_addloss'] = []


       # self.metrics_history['V_alpha_param'] = [float(self.V.alpha.detach().numpy())]

        self.hits = 0
        self.u_pts = u_pts
        u = self.generator['train'].get_examples()[0]  # mock u
        self.batch_size = len(u)
        self.sofT_pts = int(self.batch_size/self.u_pts)
        
        self.contrastive_weights=contrastive_weights
        if self.contrastive_weights is not None:
            print('CONTRASTIVE WEIGHTS Active!')

        # try:
        #     self.lowest_loss
        # except NameError:
        #     self.lowest_loss = min(self.metrics_history['train_loss'])

        if phim is not None:
            self.phim = phim
        else:
            self.phim = 0.55

        phi_mock = np.linspace(0,2.5,10000)
        Vth = V_or(phi_mock, phim=self.phim)
        DVth = DV_or(phi_mock, phim=self.phim)
        DDVth = DDV_or(phi_mock, phim=self.phim)

        self.Vth_min = min(Vth)
        indx_min = np.argmin(Vth)
        self.DVth_min = DVth[indx_min]
        self.DDVth_min = DDVth[indx_min]

        print('Vth_min = ', self.Vth_min, ' at phi = ', phi_mock[indx_min])
        print('DVth_min = ', self.DVth_min)
        print('DDVth_min = ', self.DDVth_min)

    def _set_loss_fn(self, criterion):
        pass

    def loss_fn(self,r,f,x):
        #print('shape r', r.shape)
        
        if self.contrastive_weights is not None:

           # # print('f 5 shape', f[5].shape)
           #  print('phi(0)', f[5][-71::].T) 
            # we confirm that f and r are ordered like [f_{u=1}^{S1},...,f_{u=1}^{S70},...,f_{u=0}^{S1},...,f_{u=0}^{S70}]
           #  print('A(1)', f[4][:71:].T)                 ######### 70 times ########, ...,   ######### 70 times ########
           #  print('A(0)', f[4][-71::].T)                ############################# 48 times ########################
            
            w0 = torch.tensor(self.contrastive_weights).T # shape (70, 7)
            w = w0.repeat(self.u_pts, 1)  # shape (70*48=3660, 7)
          #  print('th w (3360,7) ; real',w.shape)
            #print('r shape ;',r.shape)   # r has shape (3360, 7) <-- (70, 48, 7)
            
            loss_r2 = (w * r**2).mean()
        else:
            loss_r2 = (r**2).mean() 

        
        self.metrics_history['r2_loss'].append(loss_r2.cpu().detach().item())
        self.metrics_history['phi_max'].append(f[5][99].cpu().detach().item())
     #   self.metrics_history['V_alpha_param'].append(float(self.V.alpha.detach().numpy()))
        return loss_r2
    
    def additional_loss(self,r,f,x):


        ####### Force SB solutions to have phi_H above phi_FV (called here phi_c) obtained by the FB training #######

        if self.above_phi_c_coef != 0:

            # # OPTION 1
            p = f[5].reshape(self.u_pts, self.sofT_pts)   #shape (48,70)
            pH = p[0,:]
            p0 = p[-1,:]

            phi_c_orig = torch.tensor([self.V.phi_c])


            if self.indx_init_SB != 0:
                pH_FB = (pH[:self.indx_init_SB]).reshape(-1,1) 
                b0 = torch.ones_like(pH_FB) * phi_c_orig

            pH_SB = (pH[self.indx_init_SB::]).reshape(-1,1)  # from the index of the start of the second branch to the end
            a0 = torch.ones_like(pH_SB) * phi_c_orig   # same shape as pH

            steep_coeff = 100.0

            loss_terms_1 = torch.sigmoid(steep_coeff * (a0 - pH_SB))

            if self.indx_init_SB != 0:
                loss_terms_2 = torch.sigmoid(steep_coeff * (pH_FB - b0))
            else:
                loss_terms_2 = torch.tensor([0.0])

            addloss_above_phi_c = loss_terms_1.mean() + loss_terms_2.mean()

        else:
            addloss_above_phi_c = torch.tensor([0.0])


    ########## Force monotonic ordering in phi_Z(u) for every u #######
        if self.mono_coef != 0 :

            u0 = x[0].reshape(self.sofT_pts, self.u_pts) 
    
            if self.global_epoch == 0:
                print(u0[:71,0:49])
    
            p = f[5].reshape(self.u_pts, self.sofT_pts)   #shape (48,70)
            z = torch.zeros(self.u_pts, self.sofT_pts-1)
            a = self.steep_step
    
            z = torch.relu(p[:, :-1] - p[:, 1:])
    
    
            monotonic_addloss = z.mean()
        else:
            monotonic_addloss = torch.tensor([0.0])



        if self.order_phi_H_coef != 0:

            u0 = x[0].reshape(self.u_pts, self.sofT_pts) 
        
            p = f[5].reshape(self.u_pts, self.sofT_pts)   #shape (48,70)
            pH = p[0,:]  # u=1 is index 0


            # fig = plt.figure()
            # for i in range(self.sofT_pts):
            #     plt.plot(u0[:,i].cpu().detach().numpy(), p[:,i].cpu().detach().numpy())
            #     plt.scatter(u0[0,i].cpu().detach().numpy(), pH[i].cpu().detach().numpy(), color='k', s=5)
            # plt.show()

            a = self.steep_step

            # print(pH.shape)
            # print(pH[:-1].shape)
            # print(p[1:].shape)
    
            # val = torch.sigmoid(a*(pH[:-1] - pH[1:]))    # phi_{n} - phi_{n+1} < 0
            val = torch.relu(pH[:-1] - pH[1:])

            # print('diff val', val)
            # print(torch.mean(val))

            phi_H_ordering_addloss =  torch.mean(val)

            if self.global_epoch==0:
                print('should be 0, phi(u=0) fro every S(T)', p[-1,:])

            # if self.global_epoch % 100==0:
            #     print('addloss', phi_H_ordering_addloss)

        else:
            phi_H_ordering_addloss = torch.tensor([0.0])


        ########## Force the potential to the FB by addloss - this cannot be used if matching is done #####

        if self.FB_coef != 0.0:
    
            phi_c = self.V.phi_c
            phi = torch.linspace(0, phi_c.item(), 1000, requires_grad = True).reshape(-1,1)
            #print('grad phi ?', phi.requires_grad)
    
            V1 = self.V.First_branch_V_NN(phi).detach()
            V2 = self.V(phi)
    
            add_loss_FB = ((V1 - V2)**2).mean()
        else:
            add_loss_FB = torch.tensor([0.0])

        ############## Force BC for the potential #############
        if self.force_V0_coef != 0:
            cosa=f[5][0].reshape(-1,1) * 0.0
            #print(cosa)
            V = self.V(cosa)
            DV = diff(self.V(cosa), cosa, shape_check=False)
            DDV = diff(diff(self.V(cosa),cosa,shape_check=False),cosa,shape_check=False)
            add_loss = (3 + DDV[-1])**2 + (0 - DV[-1])**2 + (3 + V[-1])**2
            
            # phiH = f[5][:99]
            # diffs = phiH[1:]-phiH[:-1]
            # add_loss = torch.sum(torch.relu(-diffs))

            # # Force derivatives of the residuals to be zero #
            # derivs = []
            # res = torch.sum(r,dim =1)
            # derivs = diff(res,x[0],order = 1,shape_check = False)**2
            # add_loss = derivs.mean()

            V0_addloss = add_loss
        else:
            V0_addloss = torch.tensor([0.0])

        ########### Inform about the position of the V minimum ##########

        if self.force_V_min_coef != 0:

            p = f[5].reshape(self.u_pts, self.sofT_pts)  #shape (48,70)
            pH = p[0,:]
            #p0 = p[-1,:]
            p_last = pH[-1].reshape(-1,1)

            Vth_min = torch.tensor(self.Vth_min)
            Vnn_min = self.V(p_last)
            if self.global_epoch == 0:
                print('Vmin (u=1, (S,T)=(0,0)) = ', Vnn_min.item(), ' at phi = ', p_last.item(), ' with Vth_min = ', Vth_min.item())
                print('--------------------------------------------------------------------------')

            V_min_addloss = ((Vnn_min - Vth_min)**2)

        else:
            V_min_addloss = torch.tensor([0.0])

        if self.force_DV_min_coef != 0 or self.force_DDV_min_coef != 0:

            p = f[5].reshape(self.u_pts, self.sofT_pts)  #shape (48,70)
            pH = p[0,:]
            #p0 = p[-1,:]
            p_last = pH[-1].reshape(-1,1)
            DVmin_nn = diff(self.V(p_last), p_last, shape_check=False)
            DDVmin_nn = diff(self.V(p_last),p_last,shape_check=False, order=2)

            # DVmin_th = DV_or(p_last, phim=self.phim)   # these are wrong because we are computing the true derivative at the wrong phi value
            # DDVmin_th = DDV_or(p_last, phim=self.phim)

            DVmin_th = torch.tensor(self.DVth_min)
            DDVmin_th = torch.tensor(self.DDVth_min)

            DV_min_addloss = (DVmin_nn - DVmin_th)**2
            DDV_min_addloss = (DDVmin_nn - DDVmin_th)**2

        else:
            DV_min_addloss = torch.tensor([0.0])
            DDV_min_addloss = torch.tensor([0.0])

        ############ Staging the losses ###############

        x = self.global_epoch

        a = 1e-4     # Slope of the tanh function for the staging

        y = torch.sigmoid(a*(x - self.x0))

        force_V_min_coef = self.force_V_min_coef * y
        force_DV_min_coef = self.force_DV_min_coef  * y
        force_DDV_min_coef = self.force_DDV_min_coef  * y
        force_V0_coef = self.force_V0_coef * y

        if self.global_epoch == 0:
            print('Vmin coef (staged):', y, force_V_min_coef)
            print('DV, DDV_min coef (staged):', y, force_DV_min_coef, force_DDV_min_coef)
            print('V0 coef (staged):', y, force_V0_coef)

        ###### Compute the full loss ###########


        if self.global_epoch == 0:
            add_loss_FB_val = self.FB_coef * 1 * add_loss_FB
            add_loss = self.mono_coef* monotonic_addloss + add_loss_FB_val + self.above_phi_c_coef * addloss_above_phi_c + \
                        force_V_min_coef * V_min_addloss + force_DV_min_coef * DV_min_addloss + force_DDV_min_coef * DDV_min_addloss + force_V0_coef * V0_addloss + \
                        self.order_phi_H_coef * phi_H_ordering_addloss

        else:
            # print(self.lowest_loss)
            add_loss_FB_val = self.FB_coef * min(self.metrics_history['train_loss']) * add_loss_FB
            add_loss = self.mono_coef* monotonic_addloss + add_loss_FB_val + self.above_phi_c_coef * addloss_above_phi_c + \
                        force_V_min_coef * V_min_addloss + force_DV_min_coef * DV_min_addloss + force_DDV_min_coef * DDV_min_addloss + force_V0_coef * V0_addloss + \
                        self.order_phi_H_coef * phi_H_ordering_addloss

        self.metrics_history['monotonic_phi_add_loss'].append((self.mono_coef* monotonic_addloss).cpu().detach().item())
        self.metrics_history['FB_addloss'].append((add_loss_FB_val).cpu().detach().item())
        self.metrics_history['above_phi_c_addloss'].append((self.above_phi_c_coef * addloss_above_phi_c).cpu().detach().item())
        self.metrics_history['V_min_addloss'].append((force_V_min_coef * V_min_addloss).cpu().detach().item())
        # self.metrics_history['DV_DDV_min_addloss'].append((force_DV_min_coef * DV_min_addloss + force_DDV_min_coef * DDV_min_addloss).cpu().detach().item())

        self.metrics_history['DV_min_addloss'].append((force_DV_min_coef * DV_min_addloss).cpu().detach().item())
        self.metrics_history['DDV_min_addloss'].append((force_DDV_min_coef * DDV_min_addloss).cpu().detach().item())

        self.metrics_history['V0_addloss'].append((force_V0_coef * V0_addloss).cpu().detach().item())
        self.metrics_history['phiH_ordering_addloss'].append((self.order_phi_H_coef * phi_H_ordering_addloss).cpu().detach().item())
        self.metrics_history['add_loss'].append((add_loss).cpu().detach().item())

        return add_loss

    def _update_best(self, key):
        r"""Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history['r2_loss'][-1]
        thresh = 1000
        #print(self.count)
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.count = 0
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)
        # else:
        #     self.count += 1
        #     condition_lossstuck = (abs(min(self.metrics_history['r2_loss'][-10:])-self.lowest_loss)/self.lowest_loss)<0.5e-1
        #     #print(abs(min(self.metrics_history['r2_loss'][-10:])-self.lowest_loss)/self.lowest_loss)
        #     for g in self.optimizer.param_groups:
        #         condition_lowLR = abs(g['lr'])<self.lowest_loss*1e-1
        #     both_conditions = condition_lossstuck or condition_lowLR  
        #     if self.count>=thresh and both_conditions and False:
        #         for g in self.optimizer.param_groups:
        #             print('Learning rate re-adjusted from ' + str(g['lr']) + ' to ' + str(5*g['lr']))
        #             g['lr'] = 2*g['lr']
        #             self.count = 0
        #             self.hits += 1

    def fit(self, max_epochs, callbacks=(), tqdm_file='default', **kwargs):
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



    

# DEFINE THE WHOLE RUTINE
class NNholo():

    def __init__(self, data_path, saving_path, load_path = None,  new_FB_path=None, contrastive_weights=None, n_points = 69, T_min=0.001, u_pts = 48,\
                 split_branches = None, init_pt_curve = 55, end_pt_curve = None, delta = 0.0, curriculum = 1.0, nets_loc_var = 4, \
                 add_index = True, step = 1, solver_nets = [64,64,64], V_nets = [32,32,32,32], V_nets_second_branch = None, \
                 sampling_method = 'chebyshev2-noisy', above_phi_c_coef = 0.0, order_match_V_derivative = 3, \
                 mono_phi_coef = 1e-5, first_branch_coef = 0, force_V_min_coef=0, force_DV_min_coef=0, force_DDV_min_coef=0, force_V0_coef=0, order_phi_H_coef = 0, \
                 staging_addloss_epoch = 100_000, steep_step = 100, phim = None, seed = None):

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
            print('Using random seed:', seed)
        else:
            print('Using preselected seed:', seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # For CUDA (if using GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        self.delta = delta
        self.curriculum = curriculum
        self.path = saving_path
        self.data_path = data_path

        self.init_pt_curve = init_pt_curve
        self.end_pt_curve = end_pt_curve
        self.u_pts = u_pts

        self.mono_phi_coef = mono_phi_coef
        self.FB_coef = first_branch_coef
        self.steep_step = steep_step
        self.phim = phim

        self.staging_addloss_epoch = staging_addloss_epoch

        
        if contrastive_weights is not None:
            self.contrastive_w = np.load(contrastive_weights)
        else:
            self.contrastive_w = None

        suffix_data_path = Path(data_path).suffix
        
        if suffix_data_path == ".csv":
            df_data = pd.read_csv(data_path, header=None).values
            print('File is .csv')
            S_true_1= torch.tensor(df_data[init_pt_curve:100:1,1])
            T_true_1= torch.tensor(df_data[init_pt_curve:100:1,0])

            #S_true_2= torch.tensor(df_data[71:100:5,1])4
            #T_true_2= torch.tensor(df_data[71:100:5,0])

            S_true_3= torch.tensor(df_data[101:200:6,1])
            T_true_3= torch.tensor(df_data[101:200:6,0])

            S_true_4= torch.tensor(df_data[201::30,1])
            T_true_4= torch.tensor(df_data[201::30,0])

            S_true_lowest = torch.tensor([0])
            T_true_lowest = torch.tensor([0])
            
            self.S_true=torch.cat([S_true_1, S_true_3, S_true_4, S_true_lowest],dim=0)
            self.T_true=torch.cat([T_true_1, T_true_3, T_true_4, T_true_lowest],dim=0)
            
        if suffix_data_path == ".txt":
            df_data = pd.read_csv(data_path, sep=" ", header=None).values
            split_data = np.array([row[0].split('\t') for row in df_data], dtype=np.float64)

            print('File is .txt')
            S_all = torch.tensor(split_data[::,1])
            T_all = torch.tensor(split_data[::,0])
            
            
            self.S_true_all = torch.cat([S_all],dim = 0)[::step]
            self.T_true_all = torch.cat([T_all],dim = 0)[::step]

        S = self.S_true_all.cpu().detach().numpy()[0::]
        T = self.T_true_all.cpu().detach().numpy()[0::]
        fig = plt.figure(figsize=(14,5))
        plt.scatter(T,S/T**3, s=30)
        
        alpha = S[-2]/(T[-2])**3
        plt.axhline(alpha,color='k', linewidth=1, linestyle='dotted')
        
        n_add_pts = 10
        T_min = T_min
        for i in range(n_add_pts):
            x = np.linspace(T[-2],T_min, n_add_pts+1)
            y = alpha* x**3
        #print(T)
        #print(x)
        new_T = np.concatenate([T[0:-1:],x[1::]])
        new_S = np.concatenate([S[0:-1:],y[1::]])
        
        plt.scatter(new_T, new_S/new_T**3, s=6, color='r')
        plt.show()


        for i in range(len(new_T)-1):
            delta = new_T[i+1] - new_T[i]
            if delta <=0:
                self.indx_branch_1 = i
            else:
                print(self.indx_branch_1)
                break
                

        for i in range(len(new_T[self.indx_branch_1::])-2):
            delta = new_T[i+self.indx_branch_1+2] - new_T[i+self.indx_branch_1+1]
            if delta >=0:
                self.indx_branch_2 = i + self.indx_branch_1
            else:
                print(self.indx_branch_2)

                break

        self.S_true_whole = torch.tensor(new_S)
        self.T_true_whole = torch.tensor(new_T)
                
        if split_branches==1:
            self.S_true_all = torch.tensor(new_S)[init_pt_curve:self.indx_branch_1:step]
            self.T_true_all = torch.tensor(new_T)[init_pt_curve:self.indx_branch_1:step]   
        elif split_branches==2:
            self.S_true_all = torch.tensor(new_S)[self.indx_branch_1:self.indx_branch_2:step]
            self.T_true_all = torch.tensor(new_T)[self.indx_branch_1:self.indx_branch_2:step]
        elif split_branches==3:
            self.S_true_all = torch.tensor(new_S)[self.indx_branch_2::step]
            self.T_true_all = torch.tensor(new_T)[self.indx_branch_2::step]
        else:
            if self.end_pt_curve is not None:
                self.S_true_all = torch.tensor(new_S)[self.init_pt_curve:self.end_pt_curve:step]
                self.T_true_all = torch.tensor(new_T)[self.init_pt_curve:self.end_pt_curve:step]
            else:
                self.S_true_all = torch.tensor(new_S)[self.init_pt_curve::step]
                self.T_true_all = torch.tensor(new_T)[self.init_pt_curve::step]

        # indx_init_SB = -(self.init_pt_curve-(self.indx_branch_1+2))
        # print('indx_init_SB', indx_init_SB)

        self.Sigma_uh_all = (self.S_true_all/np.pi)**(1/3)
        self.Va_uh_all = (-self.T_true_all*4*np.pi)
        
        self.SofT_gen = torch.cat((self.Sigma_uh_all.reshape(-1,1),self.Va_uh_all.reshape(-1,1)),dim = 1)

        # Affine parameter and uniform sampling routine #

        # Define Z_list with uniform sampling
        Z_list = torch.arange(0, len(self.Sigma_uh_all.cpu().detach().numpy()), 1)  # Use arange instead of deprecated range

        # Interpolation functions
        T_of_Z = CubicSpline(Z_list.cpu().detach().numpy(), self.Va_uh_all.cpu().detach().numpy())
        S_of_Z = CubicSpline(Z_list.cpu().detach().numpy(), self.Sigma_uh_all.cpu().detach().numpy())
        
        # Tangent vector norm as a lambda function
        tang_vector_norm = lambda x: np.sqrt(T_of_Z(x, nu=1) ** 2 + S_of_Z(x, nu=1) ** 2)

        # Affine parameter function
        affine_parameter = lambda Z: quad(tang_vector_norm, 0, Z, epsabs=1e-18, epsrel=1e-18, limit=5000)

        # Compute affine parameter values
        s_list = []
        e_list = []
        for Z in Z_list.cpu().detach().numpy():
            s, e = affine_parameter(Z)
            s_list.append(s)
            e_list.append(e)

        s_list = torch.tensor(s_list)

        # Interpolate using affine parameter
        self.T_of_s = CubicSpline(s_list.cpu().detach().numpy(), self.Va_uh_all.cpu().detach().numpy())
        self.S_of_s = CubicSpline(s_list.cpu().detach().numpy(), self.Sigma_uh_all.cpu().detach().numpy())

        # Generate uniform sampling in affine parameter space
        s_sampling = np.linspace(0, max(s_list.cpu().detach().numpy()), n_points)
        
        # Compute new sampled values
        self.Va_uh_all = torch.tensor(self.T_of_s(s_sampling))
        self.Sigma_uh_all = torch.tensor(self.S_of_s(s_sampling))

        self.S_true = torch.tensor((self.Sigma_uh_all.cpu().detach().numpy())**3 * np.pi)
        self.T_true = torch.tensor(- (self.Va_uh_all.cpu().detach().numpy())/(4*np.pi))


        self.indx_init_SB = 0
        for i in range(len(self.T_true)-1):
            delta = self.T_true[i+1] - self.T_true[i]
            if delta <=0:
                self.indx_init_SB = i + 1
            else:                    
                break

        print('indx_init_SB', self.indx_init_SB)

        S_mock = self.S_true.cpu().detach().numpy()
        T_mock = self.T_true.cpu().detach().numpy()
        soverT3 = S_mock / T_mock**3

        plt.scatter(T_mock, soverT3)
        plt.scatter(T_mock[self.indx_init_SB], soverT3[self.indx_init_SB], color='red', label='2nd branch init')
        plt.legend()
        plt.show()
 
        Z = torch.tensor(s_sampling)
        self.Z = Z/torch.max(Z)
        
        print('Check device', self.Va_uh_all[0])
        
        self.SofT_gen = torch.cat((self.Sigma_uh_all.reshape(-1,1),self.Va_uh_all.reshape(-1,1)),dim = 1)
        if add_index:
            self.pg = PredefinedGenerator(self.Sigma_uh_all, self.Va_uh_all,self.Z)
        else:
            self.pg = PredefinedGenerator(self.Sigma_uh_all, self.Va_uh_all)

        self.g1 = Generator1D(u_pts, 0, self.curriculum, method=sampling_method)
        self.g2 = Generator1D(16, 0, 1, method='equally-spaced')
        self.train_generator =  MeshGenerator(self.g1, self.pg)
        self.valid_generator =  MeshGenerator(self.g2, self.pg)


        if load_path is not None:
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            master_dict = torch.load(load_path, map_location=device, weights_only=False)
            if new_FB_path is not None:
                master_dict_new_FB = torch.load(new_FB_path, map_location=device, weights_only=False)


            self.nets_arch_FB = master_dict['nets_arch']
            self.V_arch_FB = master_dict['V_arch']
            if 'nets_arch_SB' in master_dict:
                self.nets_arch_SB = master_dict['nets_arch_SB']
            else:
                self.nets_arch_SB = solver_nets
            if 'V_arch_SB' in master_dict:
                self.V_arch_SB = master_dict['V_arch_SB']
            else:
                self.V_arch_SB = V_nets

        
            self.V1 = CustomNN(n_input_units = 1, hidden_units = self.V_arch_FB ,actv = nn.SiLU, n_output_units = 1)


            self.V1.load_state_dict(master_dict['state_dict_V'])
            # self.V1.load_state_dict(master_dict_new_FB['state_dict_V'])

            phi_mock = torch.linspace(0,2.5, 1000, requires_grad=True).reshape(-1,1)
            V1_mock = self.V1(phi_mock)
            DV1_mock = diff(self.V1(phi_mock), phi_mock, shape_check=False)
            DDV1_mock = diff(self.V1(phi_mock), phi_mock, order=2, shape_check=False)
            print('From self.V1: ','V1(0)=', V1_mock[0].item(), ' DV1(0)=', DV1_mock[0].item(), ' DDV1(0)=', DDV1_mock[0].item())


            # self.phi_c = master_dict['phi_c']
            # print('phi_c', self.phi_c)

            #### CHANGING THIS (above substituted by below):
            if new_FB_path is not None:
                self.V1.load_state_dict(master_dict_new_FB['state_dict_V'])
                print('Loaded new FB state dict for V1')

                if 'phi_c' in master_dict_new_FB:
                    if master_dict['phi_c'] != master_dict_new_FB['phi_c']:
                        print('Different phi_c from FB and SB NNs -- Taking FB value!')

                    self.phi_c = master_dict_new_FB['phi_c']
                    # self.phi_c = master_dict['phi_c']

            else:
                self.V1.load_state_dict(master_dict['state_dict_V'])
                self.phi_c = master_dict['phi_c']


            if self.V_arch_SB:
                self.V = Second_Branch_V(n_input_units = 1, hidden_units = self.V_arch_SB ,actv = nn.SiLU, n_output_units = 1, \
                                         First_branch_V_NN = self.V1, phi_c = self.phi_c, order = order_match_V_derivative)
            else:
                self.V = Second_Branch_V(n_input_units = 1, hidden_units = V_nets ,actv = nn.SiLU, n_output_units = 1, \
                             First_branch_V_NN = self.V1, phi_c = self.phi_c, order = order_match_V_derivative)
                
            if 'state_dict_second_branch_V' in master_dict:
                self.V.load_state_dict(master_dict['state_dict_second_branch_V'])
                print('Loaded second branch V network parameters')
                
                if new_FB_path is not None:
                    self.V.First_branch_V_NN.load_state_dict(master_dict_new_FB['state_dict_V'])
                    print('loaded new FB (forced V(0), DV(0), DDV(0) FB net)')
                    # self.V.First_branch_V_NN.load_state_dict(master_dict['state_dict_V'])

            phi_mock = torch.linspace(0,2.5, 1000, requires_grad=True).reshape(-1,1)
            V1_mock = self.V.First_branch_V_NN(phi_mock)
            DV1_mock = diff(self.V1(phi_mock), phi_mock, shape_check=False)
            DDV1_mock = diff(self.V1(phi_mock), phi_mock, order=2, shape_check=False)
            print('From self.V.FB: ','V1(0)=', V1_mock[0].item(), ' DV1(0)=', DV1_mock[0].item(), ' DDV1(0)=', DDV1_mock[0].item())

        else:
            self.V = CustomNN(n_input_units = 1, hidden_units = V_nets ,actv = nn.SiLU, n_output_units = 1)


        self.conditions = [
    NoCondition(),  # no condition on Vs
    BundleIVP(1, None, bundle_param_lookup=dict(u_0=1)), #condition on Va = -4 pi T
    BundleIVP(0, 1),   # Vphi(0) ==1
    BundleDirichletBVP(0, 1, 1, None, bundle_param_lookup=dict(u_1=0)),  # Sigma_{u=0} = 1, Sigma_{u=1}=(S/pi)**(1/3)
    BundleDirichletBVP(0, 1, 1, 0),   # A (0) == 1  A(1)=0
    BundleIVP(0, 0),  #phi(0)=0 #BundleDirichletBVP(0, 0,1, phi_yago[-1])#

]
        if load_path is not None:
            solver_nets = self.nets_arch_SB

        self.nets = [CustomNets(n_input_units=4, hidden_units=solver_nets, n_output_units = 1,
                        actv = nn.Tanh, loc_var = nets_loc_var) for _ in range(6)]
        
        # if 'state_dict_second_branch_V' in master_dict:  # Only continue training second branch (not in change from first to second)
        #     for i in range(len(self.nets)):
        #         self.nets[i].load_state_dict(master_dict['state_dict_solver'][i])
        #     print('Loaded second branch solver nets state')

        
        self.adam = torch.optim.Adam([p for net in self.nets + [self.V] for p in net.parameters()], \
                        lr=1e-3)#,  betas=(0.9, 0.99))
        
        if load_path is not None:
            # Can only load optimizer for continue training the second branch (or first branch, although for training the first branch
            # there is already a separate code), not when loading a first branch and begin training the second branch.
            # This is why we put the following 'if' :
            if 'state_dict_second_branch_V' in master_dict:
                self.adam.load_state_dict(master_dict['optimizer'])
                print('Loaded Optimizer State')

        
        self.lbfgs = torch.optim.LBFGS(OrderedSet([p for net in self.nets + [self.V] for p in net.parameters()]), \
                        lr=1e-2)
        
        self.solver = CustomBundleSolver1D( ode_system=self.equations,
                                            conditions=self.conditions,
                                            t_min=self.delta,
                                            t_max=1,
                                            train_generator=self.train_generator,
                                            valid_generator=self.valid_generator,
                                            optimizer=self.adam,
                                            nets=self.nets,
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V,
                                            contrastive_weights = self.contrastive_w,
                                            u_pts= self.u_pts,
                                            mono_phi_coef = self.mono_phi_coef,
                                            steep_step = self.steep_step,
                                            FB_coef = self.FB_coef,
                                            indx_init_SB=self.indx_init_SB,
                                            above_phi_c_coef = above_phi_c_coef,
                                            force_V_min_coef = force_V_min_coef,
                                            force_DV_min_coef = force_DV_min_coef,
                                            force_DDV_min_coef = force_DDV_min_coef,
                                            force_V0_coef = force_V0_coef,
                                            order_phi_H_coef = order_phi_H_coef,
                                            phim=self.phim,
                                            staging_addloss_epoch = self.staging_addloss_epoch,
                                        )
        

        # Again, this should be if it is a continuation of the second branch training, not if it is the transition
        # from first branch to second branch.
        if load_path is not None and 'state_dict_second_branch_V' in master_dict:  

            if 'r2_loss' in master_dict:  
                self.solver.metrics_history['r2_loss'] = master_dict['loss']

            if 'train_loss' in master_dict:  
                self.solver.metrics_history['train_loss'] = master_dict['train_loss']

            if 'add_loss' in master_dict:  # Check if it exists for backward compatibility
                self.solver.metrics_history['add_loss'] = master_dict['add_loss']

            if 'monotonic_phi_add_loss' in master_dict: 
                self.solver.metrics_history['monotonic_phi_add_loss'] = master_dict['monotonic_phi_add_loss']

            if 'FB_addloss' in master_dict: 
                self.solver.metrics_history['FB_addloss'] = master_dict['FB_addloss'] 

            if 'above_phi_c_addloss' in master_dict: 
                self.solver.metrics_history['above_phi_c_addloss'] = master_dict['above_phi_c_addloss']  

            if 'V_min_addloss' in master_dict:  
                self.solver.metrics_history['V_min_addloss'] = master_dict['V_min_addloss']  

            if 'DV_DDV_min_addloss' in master_dict: 
                self.solver.metrics_history['DV_DDV_min_addloss'] = master_dict['DV_DDV_min_addloss']  


            if 'DV_min_addloss' in master_dict: 
                self.solver.metrics_history['DV_min_addloss'] = master_dict['DV_min_addloss']  
            if 'DDV_min_addloss' in master_dict: 
                self.solver.metrics_history['DDV_min_addloss'] = master_dict['DDV_min_addloss']  


            if 'V0_addloss' in master_dict:  
                self.solver.metrics_history['V0_addloss'] = master_dict['V0_addloss']
            if 'phiH_ordering_addloss' in master_dict:  
                self.solver.metrics_history['phiH_ordering_addloss'] = master_dict['phiH_ordering_addloss']  


            print('Loaded previous addlosses from SB training')


                

        self.solver.best_nets = np.ones_like(self.solver.nets)

        if load_path is not None and 'state_dict_second_branch_V' in master_dict:  # Only continue training second branch (not in change from first to second)

            for i in range(len(self.solver.nets)):
                self.solver.nets[i].load_state_dict(master_dict['state_dict_solver'][i])
                if master_dict['state_best_nets'][i] != None:
                    print('best nets not None')
                    self.solver.best_nets[i] = copy.deepcopy(self.solver.nets[i])
                    self.solver.best_nets[i].load_state_dict(master_dict['state_best_nets'][i])
            print('Loaded second branch solver nets state')

        
 


    def sofT_curve(self):
        
        print('S_min: ', min(self.S_true))
        print('Length of input s(T) curve: ',  self.S_true.shape)

        #print('(S*,T*) = ', '(',S_h,',', T_h,')')
        #[print('(S*,T*)_%i'%(i+1) ,'= ', '(',S_yago[i],',', T_yago[i],')') for i in range(len(S_yago))]
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        ax[0].scatter(self.T_true.cpu().detach().numpy(), self.S_true.cpu().detach().numpy())

        soverT3_x = self.T_true
        soverT3_y = self.S_true/(self.T_true**3)

        ax[1].scatter(soverT3_x.cpu().detach().numpy(), soverT3_y.cpu().detach().numpy())
        ax[2].scatter(self.Va_uh_all.cpu().detach().numpy(), self.Sigma_uh_all.cpu().detach().numpy(), s=10)
        #plt.scatter((self.T_true),(self.S_true), color='k',s=15, label='true')
        #plt.xlabel('T')
        #plt.ylabel('s')
        plt.title(f'S_true shape: {len(self.S_true.cpu())}, Max T: {"{:.2f}".format(max(self.T_true.cpu()))}')
        ax[0].set_xlabel('T')
        ax[0].set_ylabel('s')
        ax[1].set_xlabel('T')
        ax[1].set_ylabel('$s/T^3$')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[2].set_xlabel('Va_uh')
        ax[2].set_ylabel('Sigma_uh')
        fig.savefig(f'{self.path}/sofT.pdf')
        plt.show() 
        
    def update_generator(self, curriculum = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, 0, curriculum, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
        
    def update_optimizer(self, lr = None):
        print('inside beg')
        if lr == None:
            for g in self.adam.param_groups:
                print('Actual learning rate: ', g['lr'])
                print(g['lr'])
        else:
            print('indised else')
            for g in self.adam.param_groups:
                g['lr'] = lr
                print('Learning rate updated to: ', g['lr'])
                
        print('inside end')
        
        
    def set_curriculum(self, start = 0.0, end = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, start, end, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
    
    def equations(self, Vs, Va, Vp, Sigma, A, phi, u):

        # create the derivative of the V wrt to phi
        VF = diff(self.V(phi), phi, shape_check= False)

        ORIGP_FLAG = 0

        # the equations
        eq1 = Vs - diff(Sigma, u, order=1)
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2

        eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* self.V(phi)  \
                                    + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                    + A * (8 * Sigma - 6 * u * Vs)



        eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
            + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)

        # eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1)/torch.maximum(Vp,torch.tensor([1e-3])) -\
        #     Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG*DV_or(phi))/torch.maximum(Vp,torch.tensor([1e-3])) \
        #     +  (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)

        eq7 =  (u * Vs-Sigma) * \
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                2 * ((1-ORIGP_FLAG)*self.V(phi) + ORIGP_FLAG*V_or(phi)))
        #eq8 = 3 * u**2 * A * Vs * Vp + Sigma * (-VF + u * (Vp * (u * Va - 3 * A) + u * A * diff(Vp, u, order=1)))

        return [eq1, eq2, eq3, eq4 , eq5, eq6/torch.maximum(Vp,torch.tensor([1e-3])), eq7]#, eq8/Vp]
    

    def get_loss(self):

        residuals = self.get_residuals()
        batch = [v.reshape(-1, 1) for v in self.valid_generator.get_examples()]
        funcs = [self.solver.compute_func_val(a, b, *batch) for a, b in zip(self.solver.nets, self.solver.conditions)]
        if IN_COLAB:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch).cpu().detach().cpu().numpy()

        else:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch).cpu().detach().numpy()
        
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
    
    def plot_residuals(self, color=None, xlabel= 'epochs', ylabel = r'$\log_{10}\mathcal{L}$', fontsize = 14, \
                  figsize=(8,6), thick=0.8, left_x_lim=0):
        
        fig1 = plt.figure(figsize=figsize)
        index = torch.linspace(0,self.Va_uh_all.shape[0]-1,self.Va_uh_all.shape[0]).cpu().detach().numpy()
        u = self.g1.get_examples().cpu().detach().numpy()
        U,IND = torch.meshgrid(torch.tensor(u),torch.tensor(index))
        RES = []
        phi_solution = []
        A_solution = []
        Sigma_solution = []
        for i in range(len(self.Va_uh_all.cpu())):
            S = self.Sigma_uh_all[i].cpu().detach().numpy()*np.ones_like(u)
            T = self.Va_uh_all[i].cpu().detach().numpy()*np.ones_like(u)
            Z = self.Z[i].cpu().detach().numpy()*np.ones_like(u)
            #print(tf.solver.get_residuals(u,S,T))
            res = self.solver.get_residuals(u,S,T,Z,best= False)
            sol = self.solver.get_solution(best = True)
            RES.append(torch.stack(res).mean(dim = 0))
            phi_solution.append(sol(u,S,T,Z)[5].cpu().detach().numpy())
            A_solution.append(sol(u,S,T,Z)[4].cpu().detach().numpy())
            Sigma_solution.append(sol(u,S,T,Z)[3].cpu().detach().numpy())
        #print(RES)
        RES = torch.log10(torch.stack(RES)**2).t()
        plt.pcolormesh(U.cpu().detach().numpy(),IND.cpu().detach().numpy(),RES.cpu().detach().numpy(),cmap='rainbow')
        plt.colorbar(label = 'res$^2$')
        plt.xlabel('u')
        plt.ylabel('index')
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        fig1.savefig(f'{self.path}/res_2D_{trained_epochs}.pdf')
        plt.show()




    
    def plot_solutions(self, step=1, save_fig=True, best=True, legend=True):
        u = torch.linspace(0,1,250)
        var_names = ['Vs', 'Va', 'Vp', r'$\Sigma$', 'A', r'$\phi$']
        
        fig,ax = plt.subplots(2,3,figsize=(15, 10))
        
        for j in range(6):
            RES = []
            phi_solutions = []
            step = step
            for i in range(len(self.Va_uh_all)):
                S = self.Sigma_uh_all[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
                T = self.Va_uh_all[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
                Z = self.Z[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
                #print(c.solver.get_residuals(u,S,T))
                #res = c.solver.get_residuals(u,S,T,Z,best= False)
                sol = self.solver.get_solution(best = best)
                phi_sol = sol(u,S,T,Z)[j]
              #  RES.append(torch.stack(res).mean(dim = 0))
                phi_solutions.append(phi_sol.cpu().detach().numpy())
            
            num_curves = len(self.Va_uh_all)
            cmap = cm.get_cmap('rainbow', num_curves)
            norm = mcolors.Normalize(vmin=0, vmax=num_curves - 1)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            for i in range(num_curves):
                #if i == step:# == 0:
                if i % step == 0:
        
                    col = cmap(i / (num_curves - 1))
                    
                    y = u.cpu().detach().numpy()
                    z = phi_solutions[i]
                   # x = np.full_like(y, i)  # Constant z value per curve (index i)
                    if j<3:
                        ax[0,j].plot(y, z, color=col, linewidth=2, label=i)
                        ax[0,j].set_title(var_names[j])
                    else:
                        ax[1,j-3].plot(y, z, color=col, linewidth=2, label=i)
                        ax[1,j-3].set_title(var_names[j])
                        if j==5 and i==num_curves-1:
                            ax[1,j-3].axhline(self.phi_c, color='red', linestyle='dotted', label=r'$\phi_{FV}$')
        
        
        plt.xlabel('u')
        if legend == True:
            plt.legend(loc=(1.1,0.3))
        plt.show()
        if save_fig==True:
            fig.savefig(f'{self.path}/solutions_epoch_{self.solver.global_epoch}.pdf')

    
    def plot_3D_sofT(self, color=None, xlabel= 'epochs', ylabel = r'$\log_{10}\mathcal{L}$', fontsize = 14, \
                  figsize=(8,6), thick=0.8, left_x_lim=0):

        a = 0

        u = np.linspace(0.0001, 1, 100)
        S_true = self.S_true.cpu().detach().numpy()
        T_true = self.T_true.cpu().detach().numpy()
        Z_all= self.Z.cpu().detach().numpy()
        phi_h = np.ones(S_true.shape)
        solution = self.solver.get_solution(best=True)
        phi_h = np.ones_like(S_true)

        for i,S in enumerate(S_true):
            T=T_true[i]
            Sigma_v = (S/np.pi)**(1/3)
            Va_v = (-T*4*np.pi)
            Sigma_uh = Sigma_v*np.ones_like(u)
            Va_uh = Va_v*np.ones_like(u)
            Z = Z_all[i]*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh,Z, to_numpy=True)
            if a==1:
                phi_h[i] = phi.max()
            elif a==0:
                phi_h[i] = phi[-1]
            #true_phi_h[i] = phi[-1]
            #i_max = phi.argmax()
            #u_max[i] = u[i_max]
        print('Max phi_H is pt ', phi_h.argmax())
        print('Min phi_H is pt ', phi_h.argmin())

        step = 1

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        # Create the scatter plot
        S = self.S_true.cpu().detach().numpy()
        T = self.T_true.cpu().detach().numpy()
        Z = self.Z.cpu().detach().numpy()

        fontsize = 18

        scatter = ax.scatter((Z[::step]),(T[::step]), (S[::step]), c = phi_h[::step], s=10, cmap='rainbow')
        ax.scatter((Z[phi_h.argmax()]),(T[phi_h.argmax()]), (S[phi_h.argmax()]), c = 'k', s=100, marker='x', label = 'max phi_H')
        ax.scatter((Z[phi_h.argmin()]),(T[phi_h.argmin()]), (S[phi_h.argmin()]), c = 'k', s=100, marker='.', label = 'min phi_H')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label(r'$\phi_H$', fontsize = fontsize)
        colorbar.ax.yaxis.label.set_rotation(0)
        ax.set_zlabel(r'$S/\Lambda^3$', fontsize=fontsize)
        ax.set_ylabel(r'$T/\Lambda$', fontsize=fontsize)
        ax.set_xlabel('Z index')

        plt.legend()
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        fig.savefig(f'{self.path}/sofT_3D_{trained_epochs}.pdf')
        plt.show()        


    def plot_loss(self, color=None, xlabel= 'epochs', ylabel = r'$\log_{10}\mathcal{L}$', fontsize = 14, \
                  figsize=(8,6), thick=0.8, left_x_lim=0, save_fig = True, past_epochs = None, step = 1):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        trace = self.solver.metrics_history

        zoom_flag = True
        if past_epochs is None:
            past_epochs = trained_epochs
            zoom_flag = False

        fig1 = plt.figure(figsize=figsize)
        if color==None:
            plt.plot(np.log10(trace['train_loss'][-past_epochs::step]), label='train loss')

            #plt.plot(np.log10(trace['r2_loss']), label='DE loss')
            #plt.plot(np.log10(trace['add_loss']), label='Add loss')
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
        plt.legend(loc='upper right')
        #plt.tight_layout()
        plt.show()
        print('Min loss: ', min(trace['train_loss']))
        
        if save_fig==True:
            if zoom_flag is True:
                fig1.savefig(f'{self.path}/loss_epoch_{trained_epochs}_zoomed.pdf')
            else:
                fig1.savefig(f'{self.path}/loss_epoch_{trained_epochs}.pdf')


    def plot_separate_losses(self, past_epochs = None, save_fig = True):

        loss = self.solver.metrics_history

        tot_loss = np.array(loss['train_loss'])

        if past_epochs is None:
            past_epochs = len(tot_loss)

        DE_loss = np.array(loss['r2_loss'])
        above_phi_c_addloss = np.array(loss['above_phi_c_addloss'])
        pH_ordering_addloss = np.array(loss['phiH_ordering_addloss'])
        Vmin_addloss = np.array(loss['V_min_addloss'])
        DVDDV_min_addloss = np.array(loss['DV_DDV_min_addloss'])

        if self.solver.global_epoch - len(DVDDV_min_addloss) > 0:
            DVDDV_min_addloss = np.concatenate([np.zeros(self.solver.global_epoch - len(DVDDV_min_addloss)), DVDDV_min_addloss])



        fig = plt.figure(figsize=(6,6))
        plt.plot(tot_loss[-past_epochs::], label = 'tot loss', alpha = 0.5)
        plt.plot(DE_loss[-past_epochs::], label = 'DE loss', alpha = 0.5)

        if self.solver.order_phi_H_coef != 0:
            plt.plot(pH_ordering_addloss[-past_epochs::], label = r'$\phi_H$-ordering addloss', alpha = 0.5)
        if self.solver.above_phi_c_coef != 0:
            plt.plot(above_phi_c_addloss[-past_epochs::], label = r'$\phi_c$-above addloss', alpha = 0.5)
        if self.solver.force_V_min_coef != 0:
            plt.plot(Vmin_addloss[-past_epochs::], label = r'Vmin addloss', alpha = 0.5, color='purple')
        if self.solver.force_DV_DDV_min_coef != 0:
            plt.plot(DVDDV_min_addloss[-past_epochs::], label = r'$\partial^{1,2}$ Vmin addloss', alpha = 0.5, color='red')
        plt.yscale('log')
        if min(above_phi_c_addloss)<1e-8:
            plt.ylim(bottom=1e-8)
        plt.legend(loc=(1.1,0.5))
        if save_fig==True:
            fig.savefig(f'{self.path}/separate_losses_epochs_{self.solver.global_epoch}.pdf')
        plt.show()


    def plot_separate_losses_new(self, past_epochs = None, step = 1, save_fig = True):
        
            loss = self.solver.metrics_history

            if past_epochs is None:
                past_epochs = len(loss['train_loss'])

            fig = plt.figure(figsize=(15, 8))

            # relevant_keys = ['train_loss', 
            #                 'r2_loss', 
            #                 'V0_th_add_loss', 
            #                 'DV0_th_add_loss', 
            #                 'DDV0_th_add_loss', 
            #                 'V_FV_th_add_loss', 
            #                 'DV_FV_th_add_loss',
            #                 'DDV_FV_th_add_loss'
            #                 ]
            relevant_keys = loss.keys()

            for key in relevant_keys:
                plt.plot(loss[key][-past_epochs::step], label=key)

            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(loc=(1.01,0))
            plt.title('Loss Curves')
            plt.grid()
            plt.tight_layout()
            if save_fig == True:
                fig.savefig(f'{self.path}/separate_losses_epochs_{self.solver.global_epoch}.pdf', bbox_inches='tight')
            plt.show()






    def plot_additional_losses(self, save_fig = True):

        loss = self.solver.metrics_history

        length = len(loss['r2_loss'])

        step = int(0.001 * length)

        if step == 0:
            step = 1

        fig = plt.figure(figsize = (7,6))
        plt.plot(loss['r2_loss'][::step], label = 'DE')
        plt.plot(loss['V_min_addloss'][::step], label = 'Vmin')
        plt.plot(loss['DV_DDV_min_addloss'][::step], label = 'DVmin, DDVmin')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        if save_fig==True:
            fig.savefig(f'{self.path}/separate_losses_epochs_{self.solver.global_epoch}.pdf')

        plt.show()



    def plot_residuals_in_u_color(self, step=1, save_fig=True):

        u = torch.linspace(0.0,1.0,100)
        
        num_curves = len(self.Va_uh_all)
        cmap = cm.get_cmap('rainbow', num_curves)
        norm = mcolors.Normalize(vmin=0, vmax=num_curves - 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        for q,i in enumerate(zip(self.Sigma_uh_all, self.Va_uh_all)):
            Sigma_h = i[0].cpu().detach()*torch.ones_like(u)
            Va_h = i[1].cpu().detach()*torch.ones_like(u)
            Z = self.Z[q].cpu().detach()*torch.ones_like(u)
        
            if q % step == 0:
                #print(q)
                col = cmap(q / (num_curves - 1))
        
                res1 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[0].cpu().detach().numpy()
                res2 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[1].cpu().detach().numpy()
                res3 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[2].cpu().detach().numpy()
        
                res = [res1,res2,res3]
        
                ax[0].plot(u.cpu().detach().numpy(), res1 , color=col ,  alpha=0.3,label='Vs Eq1')
                ax[1].plot(u.cpu().detach().numpy(), res2 ,color=col ,  alpha=0.3, label='Va Eq2')
                ax[2].plot(u.cpu().detach().numpy(), res3 ,color=col , alpha=0.3,label='Vp Eq3')
                ax[0].set_title('Eq 1')
                ax[1].set_title('Eq 2')
                ax[2].set_title('Eq 3')
        
                if q==0:
                    plt.legend()

        for i in range(3):
            ax[i].axhline(np.sqrt(min(self.solver.metrics_history['train_loss'])), color='k', linestyle='dashed')
            ax[i].axhline(-np.sqrt(min(self.solver.metrics_history['train_loss'])), color='k', linestyle='dashed')
        
        
        fig2,ax2 = plt.subplots(2,2, figsize=(7,7))
        for q,i in enumerate(zip(self.Sigma_uh_all, self.Va_uh_all)):
        
            Sigma_h = i[0].cpu().detach()*torch.ones_like(u)
            Va_h = i[1].cpu().detach()*torch.ones_like(u)
            Z = self.Z[q].cpu().detach()*torch.ones_like(u)
        
            if q % step == 0:
                #print(q)
                col = cmap(q / (num_curves - 1))
        
                res4 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[3].cpu().detach().numpy()
                res5 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[4].cpu().detach().numpy()
                res6 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[5].cpu().detach().numpy()
                res7 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[6].cpu().detach().numpy()
                
                res = [res4,res5,res6, res7]
        
                ax2[0,0].plot(u.cpu().detach().numpy(), res4  , color=col, alpha=0.5 ,label='Eq4')
                ax2[0,1].plot(u.cpu().detach().numpy(), res5 ,   color=col, alpha=0.5, label='Eq5' )
                ax2[1,0].plot(u.cpu().detach().numpy(), res6  ,  color=col, alpha=0.5, label='Eq6' )
                ax2[1,1].plot(u.cpu().detach().numpy(), res7  ,  color=col, alpha=0.5, label='Eq7' )
        
                if q==0:
                    plt.legend()                    
        for i in range(3):
            ax[i].set_title(f'Eq {i+1}')
        
        for i in range(2):
            for j in range(2):
                if i==0:
                    ax2[i,j].set_title(f'Eq {4+j}')
                    ax2[i,j].set_xlabel('u')
                    ax2[i,j].axhline(np.sqrt(min(self.solver.metrics_history['train_loss'])), color='k', linestyle='dashed')
                    ax2[i,j].axhline(-np.sqrt(min(self.solver.metrics_history['train_loss'])), color='k', linestyle='dashed')
                if i==1:
                    ax2[i,j].set_xlabel('u')
                    ax2[i,j].set_title(f'Eq {6+j}')
                    ax2[i,j].axhline(np.sqrt(min(self.solver.metrics_history['train_loss'])), color='k', linestyle='dashed')
                    ax2[i,j].axhline(-np.sqrt(min(self.solver.metrics_history['train_loss'])), color='k', linestyle='dashed')
        
        if save_fig==True:
            fig.savefig(f'{self.path}/DE_res_1_to_3_color_epochs_{self.solver.global_epoch}.pdf')
            fig2.savefig(f'{self.path}/DE_res_4_to_7_color_epochs_{self.solver.global_epoch}.pdf')

        plt.show()

    def compute_relative_error(self, n_pts = 1000, phim_param = 0.8, save_fig = True):
        
        u = torch.linspace(0.0,1.0,1000)
        phi_solutions = []
        step = 1
        for i in range(len(self.Va_uh_all)):
            S = self.Sigma_uh_all[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
            T = self.Va_uh_all[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
            Z = self.Z[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())

            sol = self.solver.get_solution(best = True)
            phi_sol = sol(u,S,T,Z)[5]
            phi_solutions.append(phi_sol.cpu().detach().numpy())
        
        phi_max = max(np.concatenate(phi_solutions))
            
        phi_th =np.linspace(0,2.7,n_pts)
        phi_nn = torch.linspace(0,phi_max,n_pts).reshape(-1,1)
        phi_min_th = phi_th[np.argmin(V_or(phi_th, phim=phim_param))]
        
        re = 0.0
        fig = plt.figure(figsize=(5,5))
        Vnn = self.V(phi_nn).cpu().detach().numpy()
        V_nn_plot = []
        V_th_plot = []
        for i, p in enumerate(phi_nn.cpu().detach().numpy()):
            if p<=phi_min_th:
                Vth = V_or(p, phim=phim_param)[0]
            else:
                Vth = V_or(phi_min_th, phim=phim_param)
                #print('B')
            V_nn_plot.append(Vnn[i][0])
            V_th_plot.append(Vth)
            re+= (1/n_pts)* abs(Vth-Vnn[i][0])/abs(Vth)
            
        print('RE',re)
        
        plt.plot(phi_nn.cpu().detach().numpy(), V_th_plot, 'k', label = 'Th')
        plt.plot(phi_nn.cpu().detach().numpy(), V_nn_plot, 'g', label = 'NN')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$V(\phi)$')
        plt.title('RE=%.5f'%re)
        plt.legend()
        plt.show()
        if save_fig==True:
            fig.savefig(f'{self.path}/V_relative_error_epoch_{self.solver.global_epoch}.pdf')

    def plot_potential(self, phim,n_points = 100, save_fig = True, best = False):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        u = np.linspace(0.0001, 1, n_points)
        solution = self.solver.get_solution(best=True)
        phi_h = np.ones(self.S_true.shape)
        true_phi_h = np.ones(self.S_true.shape)
        u_max = np.ones(self.S_true.shape)

        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
        #    print(i,S,T)
            Sigma_v = self.Sigma_uh_all[i]
            Va_v = self.Va_uh_all[i]
            Z = self.Z[i].cpu().detach().numpy()*np.ones_like(u)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, Z,to_numpy=True)
            phi_h[i] = phi.max()
            true_phi_h[i] = phi[-1]
            i_max = phi.argmax()
            u_max[i] = u[i_max]
        print('max phi_h= ',max(phi_h))
        print(phi_h)
        # Define the domain of input phi
        phi=torch.reshape(torch.linspace(0,max(phi_h)+0.3,n_points),[n_points,1])
        # phi=torch.reshape(torch.linspace(0.0,max(phi_h),100),[100,1])
        phi = torch.Tensor(phi)
        phi.requires_grad = True
        qphi = phi.cpu().detach().numpy().reshape(-1,)
        qphi.shape

        #Vv = potential_cb.best_potential(phi) #potential_cb.best_potential(phi)
        #DVv = diff(potential_cb.best_potential(phi), phi, shape_check= False)
        #DDVv = diff(potential_cb.best_potential(phi), phi, order=2, shape_check= False)
        Vv = self.V(phi)
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
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$V(\phi)$')
        
        phi_h_th = qphi[np.where(V_or(qphi,phim)==min(V_or(qphi,phim)))]
        plt.axvline(phi_h_th, color = 'blue', linestyle = 'dashed', label=r'$\phi_{h,th}=%.2f$'%phi_h_th)
        plt.axvline(max(phi_h), label=r'$\phi_{h,NN}=$%.2f'%max(phi_h), color='orange', linestyle = 'dashed')
        plt.axhline(min(V_or(qphi, phim)), color ='blue', linestyle ='dotted', label='$V_{th}^{min}=%.2f$'%min(V_or(qphi,phim)))
        
        #print('len V_NN', len(Vv),type(Vv))
        #print('len phi: ', len(phi), type(phi))
        #print('len V_NN', len(Vv.cpu().detach().numpy()),type(Vv.cpu().detach().numpy()))
        #print('len phi: ', len(phi.detach().numpy()), type(phi.detach().numpy()))
        #print(phi.detach().numpy()[:,0])
        #print('len phi_h: ', len(phi_h))
        
        #V_nn_interp = interp1d(phi.cpu().detach().numpy()[:,0], Vv.cpu().detach().numpy()[:,0])
        #V_nn_h = V_nn_interp(phi_h)
        #plt.axhline(V_nn_h[int(np.where(phi_h==max(phi_h))[0])], color = 'orange', linestyle ='dotted', label='$V_{NN}^{min}=%.2f$'%V_nn_h[int(np.where(phi_h==max(phi_h))[0])])

        plt.ylim((min(V_or(qphi,phim))-1, -2))
        plt.xlim((0.0,max(phi_h)+0.1))
        plt.legend()
        plt.show()
        plt.close()
        
        if save_fig == True:
            fig2.savefig(f'{self.path}/V_epoch {trained_epochs}.pdf')


    def plot_potential_new(self, phim, n_points = 100, save_fig = True, best = True):

            u = torch.linspace(0,1,250)
            RES = []
            phi_solutions = []
            #V_solutions = []
            step = 1
            for i in range(len(self.Va_uh_all)):
                S = self.Sigma_uh_all[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
                T = self.Va_uh_all[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
                Z = self.Z[i].cpu().detach().numpy()*np.ones_like(u.cpu().detach().numpy())
                #print(self.solver.get_residuals(u,S,T))
                #res = self.solver.get_residuals(u,S,T,Z,best= False)
                sol = self.solver.get_solution(best = best)
                phi_sol = sol(u,S,T,Z)[5]
            #  RES.append(torch.stack(res).mean(dim = 0))
                phi_solutions.append(phi_sol.cpu().detach().numpy())

            phi_samp = np.concatenate(phi_solutions)
            # plt.hist(phi_samp)
            # plt.xlabel(r'$\phi$')
            # plt.show()

            phi_max = max(np.concatenate(phi_solutions))
            phi_last=phi_solutions[-1][-1]

            phi=torch.linspace(0,2.5,n_points).reshape(-1,1)

            V = self.V(phi)

            V_th = V_or(phi.cpu().detach().numpy(), phim=phim)
            phi_1 = np.linspace(0,0.7,n_points)
            V_th_1 = V_or(phi_1, phim=0.55)

            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()

            # Main plot
            ax.plot(phi.cpu().detach().numpy(), V.cpu().detach().numpy(), label='NN')

            ax.plot(phi.cpu().detach().numpy(), V_th, '--k', linewidth=1, label='Th')
            ax.axvline(phi_max, label='phi_h max', linestyle='dotted')
            ax.axvline(phi_last, linestyle='dashed', color='green', label='phi_h last')
            ax.axvline(phi_1[np.argmin(V_th_1)], color='k', linestyle='dashed')
            ax.axvline(self.V.phi_c, color='red', linestyle='dotted')

            # --- Zoom-in plot ---
            # --- Zoom-in plot (square inset, stretched data) ---
            ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower center', borderpad=2)

            # Define zoomed region
            x1, x2 = 0, 1.5
            y1, y2 = -5, -2.5
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y1, y2)


            # Plot inside inset
            ax_inset.plot(phi.cpu().detach().numpy(), V.cpu().detach().numpy(), label='NN')
            ax_inset.plot(phi.cpu().detach().numpy(), V_th, '--k', linewidth=1, label='Th')
            ax_inset.axvline(phi_max, linestyle='dotted')
            ax_inset.axvline(phi_last, linestyle='dashed', color='green')
            ax_inset.axvline(phi_1[np.argmin(V_th_1)], color='k', linestyle='dashed')
            ax_inset.axvline(self.V.phi_c, color='red', linestyle='dotted')
            ax_inset.tick_params(axis='both', which='major', labelsize=8)

            # Draw rectangle + connectors
            mark_inset(ax, ax_inset, loc1=2, loc2=1, fc="none", ec="black", lw=0.5)

            # Final styling
            # ax.set_xlim(right=1.5)
            ax.set_ylim(bottom = min(V_th)[0]-15, top=10)

            ax.legend(loc=(1.05,0.5))
            if save_fig==True:
                fig.savefig(f'{self.path}/V_full_epochs_{self.solver.global_epoch}.pdf')
            plt.show()
        
    def plot_colored_sofT(self, save_fig=True, colormap = 'vidris', fontsize = 14, n_fontsize = 14, \
                          dot_size=10, thick=0.8, figsize = (8,6)):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        u = np.linspace(0.0001, 1, 100)
        phi_h = np.ones(self.S_true.shape)
        solution = self.solver.get_solution(best=True)
        
        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
            Sigma_v = self.Sigma_uh_all[i]
            Va_v = self.Va_uh_all[i]
            Z = self.Z[i].cpu().detach().numpy() * np.ones_like(u)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh,Z, to_numpy=True)
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
        if save_fig==True:
            fig.savefig(f'{self.path}/colored_soft_{trained_epochs}.pdf')

        plt.show()
        
    def plot_s_over_phi_h(self, color = 'k', fontsize = 14, n_fontsize = 14, \
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

        fig.savefig(f'{self.path}/colored_sofT_epochs_{self.solver.global_epoch}.pdf')

        
    def compare_to_yago(self, fontsize = 14, legend_fontsize=14, n_fontsize=14, wspace=0.5, yago_linewidth = 3, yago_style = '--'):
        
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

            solution = self.solver.get_solution(best=True)

            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_h,  Va_h, to_numpy=True)

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
            
            fig.savefig(f'/Users/pablo/Desktop/NNholo/To run/January (2024)/Plots paper/DE solution_pt_{i+1}_(phiM=1,best).pdf')
            
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
            plt.plot(phi_th[0:cnt+1],V_th[0:cnt+1],color=col[i],label=r'$\phi_M$=%.2f'%phim)
            plt.axvline(phi_min, linestyle = 'dotted', color=col[i])
            #plt.ylim(-8,-2)
            plt.xlim(0,max(phi_mins)+0.1)
            plt.grid()
            plt.legend()
        plt.show()
        
    def plot_residuals_in_u(self, max_bound = 0.05, print_overbound = False, save_fig=False):
       ### RESIDUAL PLOTS
        # PICK ANY
        u = np.linspace(0.0001, 1, 250)
        
        if max_bound != False:
            bottom, top = -max_bound, max_bound
        
        fig = plt.figure()
        for q,i in enumerate(zip(self.Sigma_uh_all.cpu(), self.Va_uh_all.cpu())):
            Sigma_h = i[0].cpu().detach().numpy()*np.ones_like(u)
            Va_h = i[1].cpu().detach().numpy()*np.ones_like(u)
            Z = self.Z[q].cpu().detach().numpy() * np.ones_like(u)

            if q % 5 == 0:
                #print(q)
                res1 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[0].cpu().detach().numpy()
                res2 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[1].cpu().detach().numpy()
                res3 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[2].cpu().detach().numpy()
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
        
        if save_fig==True:
            fig.savefig('DE residuals 1-3.pdf')
        
        fig = plt.figure()
        for q,i in enumerate(zip(self.Sigma_uh_all.cpu(), self.Va_uh_all.cpu())):

            Sigma_h = i[0].cpu().detach().numpy()*np.ones_like(u)
            Va_h = i[1].cpu().detach().numpy()*np.ones_like(u)
            Z = self.Z[q].cpu().detach().numpy() * np.ones_like(u)
                
            if q % 5 ==0:
                #print(q)
                
                res4 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[3].cpu().detach().numpy()
                res5 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[4].cpu().detach().numpy()
                res6 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[5].cpu().detach().numpy()
                res7 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[6].cpu().detach().numpy()
                #res8 = self.solver.get_residuals(u,Sigma_h , Va_h,Z,  best=True)[7].detach().numpy()
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
            
            if save_fig==True:
                fig.savefig('DE residuals 4-7.pdf')
    def compare_to_yago2(self, fontsize = 14, legend_fontsize=14, n_fontsize=14, wspace=0.5):
        from pathlib import Path
        from matplotlib.ticker import ScalarFormatter

        colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']

        params = {'axes.titlesize': small,
                  'legend.fontsize': small,
                  'figure.figsize': (4,4),
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
        from matplotlib.ticker import ScalarFormatter
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

            solution = self.solver.get_solution(best=True)

            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_h,  Va_h, to_numpy=True)

            print('Point %i' %pt, '; phi_h_yago = %f' %phi_uh_yago[pt-1])

            ax[0].plot(u, Sigma, 'r-', label=r'$\tilde{\Sigma}_{NN}$')
            ax[0].plot(u_yago , Sigma_yago_all[pt-1], 'r-.', label=r'$\tilde{\Sigma}_{th}$',linewidth = 3)

            ax[0].plot(u, A, 'b-', label=r'$\tilde{A}_{NN}$')
            ax[0].plot(u_yago, A_yago_all[pt-1], 'b-.', label = r'$\tilde{A}_{th}$',linewidth = 3) 

            ax[0].plot(u, phi, 'g-', label=r'$\phi_{NN}$')
            ax[0].plot(u_yago, phi_yago_all[pt-1], 'g-.', label=r'$\phi_{th}$',linewidth = 3)
            #ax[0].ticklabel_format(axis='y', style='sci', scilimits=(1,1))
           
            ax[0].set_xlabel('u', fontsize = fontsize)
            ax[0].set_ylabel('ODEs solutions', fontsize = fontsize)
            ax[0].set_xlim(-0.05,1.05)
            ax[0].set_ylim(-0.05,max(max(Sigma)+0.05,max(A)+0.05,max(phi)+0.05))
            
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
            
            RMS_Sigma = np.sqrt(sum(rel_err_Sigma**2)/len(rel_err_Sigma))
            RMS_A = np.sqrt(sum(rel_err_A**2)/len(rel_err_A))
            RMS_phi = np.sqrt(sum(rel_err_phi**2)/len(rel_err_phi))
            
            ax[1].plot(u,(Sigma_yago_all[pt-1]-Sigma)**2, color= 'r', label=r'$MSE_{\Sigma}$ = %.2E' % MSE_Sigma)
            ax[1].plot(u,(A_yago_all[pt-1]-A)**2, color= 'b', label=r'$MSE_{A}$ = %.2E' % MSE_A)
            ax[1].plot(u,(phi_yago_all[pt-1]-phi)**2, color= 'g', label=r'$MSE_{\phi}$ = %.2E' % MSE_phi)
            #ax[1].ticklabel_format(axis='y', style='sci', scilimits=(1,1))
            ax[1].set_xlabel('u',fontsize = fontsize)
            ax[1].set_ylabel(r'(Theory-NN)$^2$',fontsize = fontsize)
            ax[1].set_xlim(0,1)
            ax[1].set_ylim(1e-12,max((phi_yago_all[pt-1]-phi)**2)*2)
            ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax[1].yaxis.get_offset_text().set_fontsize(n_fontsize)
            

            ax[1].legend(fontsize = legend_fontsize)
            
            title = r"$(T_{:.0f}, S_{:.0f})=({:.2f}, {:.2f})$".format(i+1,i+1,T_tt, S_tt)
            plt.suptitle(title, fontsize = fontsize)
            
            for thing in ax:
                thing.tick_params(axis='both', which='major', labelsize=n_fontsize)# Adjust the font size
            plt.subplots_adjust(wspace=wspace)  # Increase or decrease the value to adjust the spacing
            ax[1].set_yscale('log')
            plt.savefig("DE_solution_pt"+str(i+1)+"_(phiM =1,best).pdf")
            plt.show()
    def render(self):

        self.plot_loss()
        #self.plot_residuals()
        #self.plot_result()
        #self.compare_to_yago2()
        self.plot_potential(potential_cb,phim = 1.0,n_points = 5000)




    def save_results(self, path):
    
        nets_state = []
        best_nets_state = []
    
        for i in range(len(self.solver.nets)):
            nets_state.append(self.solver.nets[i].state_dict())
            best_nets_state.append(self.solver.best_nets[i].state_dict())
    
        V_net_state = self.V1.state_dict()
        V_net_state_SB = self.V.state_dict()
        
        phi_c = self.phi_c
    
        state = {'epoch': self.solver.global_epoch, 
                 'state_dict_V': V_net_state,   # first branch V state
                 'state_dict_second_branch_V': V_net_state_SB,  # second branch V state
                 # 'state_dict_first_branch_V': First_branch_V_net_state,
                 'phi_c': phi_c,
                 'nets_arch': self.nets_arch_FB,
                 'V_arch': self.V_arch_FB,
                 'nets_arch_SB': self.nets_arch_SB,
                 'V_arch_SB': self.V_arch_SB,
                # 'state_V_dense': V_dense_state,
               #  'state_V_alpha_param': V_alpha_param,
                 'state_dict_solver': nets_state,
                 'state_best_nets': best_nets_state,
                # 'state_dense_nets': dense_nets_state,
               #  'state_nets_alpha_param': nets_alpha_param,
                 'optimizer': self.adam.state_dict(), 
                 'loss': self.solver.metrics_history['r2_loss'],
                 'train_loss': self.solver.metrics_history['train_loss'],
                 'monotonic_phi_add_loss': self.solver.metrics_history['monotonic_phi_add_loss'],
                 'FB_addloss': self.solver.metrics_history['FB_addloss'],
                 'add_loss': self.solver.metrics_history['add_loss'],
                 'above_phi_c_addloss': self.solver.metrics_history['above_phi_c_addloss'],
                 'V_min_addloss': self.solver.metrics_history['V_min_addloss'],
                 'DV_min_addloss': self.solver.metrics_history['DV_min_addloss'],
                 'DDV_min_addloss': self.solver.metrics_history['DDV_min_addloss'],
                 'V0_addloss': self.solver.metrics_history['V0_addloss'],
                 'phiH_ordering_addloss': self.solver.metrics_history['phiH_ordering_addloss'],
                #  'DV_DDV_min_addloss': self.solver.metrics_history['DV_DDV_min_addloss'],
                }

        torch.save(state,path+'_epochs_' + str(int(self.solver.global_epoch)))
        print('Model succesfully saved')


    def save_results_light(self, path):
    
        nets_state = []
        best_nets_state = []
    
        for i in range(len(self.solver.nets)):
            nets_state.append(self.solver.nets[i].state_dict())
            best_nets_state.append(self.solver.best_nets[i].state_dict())
    
        V_net_state = self.V1.state_dict()
        V_net_state_SB = self.V.state_dict()
        
        phi_c = self.phi_c
    
        state = {'epoch': self.solver.global_epoch, 
                 'state_dict_V': V_net_state,   # first branch V state
                 'state_dict_second_branch_V': V_net_state_SB,  # second branch V state
                 # 'state_dict_first_branch_V': First_branch_V_net_state,
                 'phi_c': phi_c,
                 'nets_arch': self.nets_arch_FB,
                 'V_arch': self.V_arch_FB,
                 'nets_arch_SB': self.nets_arch_SB,
                 'V_arch_SB': self.V_arch_SB,
                # 'state_V_dense': V_dense_state,
               #  'state_V_alpha_param': V_alpha_param,
                 'state_dict_solver': nets_state,
                 'state_best_nets': best_nets_state,
                # 'state_dense_nets': dense_nets_state,
               #  'state_nets_alpha_param': nets_alpha_param,
                 'optimizer': self.adam.state_dict(), 
                #  'loss': self.solver.metrics_history['r2_loss'],
                #  'train_loss': self.solver.metrics_history['train_loss'],
                #  'monotonic_phi_add_loss': self.solver.metrics_history['monotonic_phi_add_loss'],
                #  'FB_addloss': self.solver.metrics_history['FB_addloss'],
                #  'add_loss': self.solver.metrics_history['add_loss'],
                #  'above_phi_c_addloss': self.solver.metrics_history['above_phi_c_addloss'],
                #  'V_min_addloss': self.solver.metrics_history['V_min_addloss'],
                #  'DV_min_addloss': self.solver.metrics_history['DV_min_addloss'],
                #  'DDV_min_addloss': self.solver.metrics_history['DDV_min_addloss'],
                #  'V0_addloss': self.solver.metrics_history['V0_addloss'],
                #  'phiH_ordering_addloss': self.solver.metrics_history['phiH_ordering_addloss'],
                #  'DV_DDV_min_addloss': self.solver.metrics_history['DV_DDV_min_addloss'],
                }

        torch.save(state, path+'_epochs_' + str(int(self.solver.global_epoch)) + '_light')
        print('Model succesfully saved')


    def load_results(self, path):
    
        # Add map_location to handle loading GPU models on CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        master_dict = torch.load(path, map_location=device, weights_only=False)
        
        self.V.load_state_dict(master_dict['state_dict_V'])
        # self.V.dense.load_state_dict(master_dict['state_V_dense'])
        # self.V.alpha = master_dict['state_V_alpha_param']
        
        self.adam.load_state_dict(master_dict['optimizer'])
        self.solver = CustomBundleSolver1D( ode_system=self.equations,
                                            conditions=self.conditions,
                                            t_min=0.0,
                                            t_max=1.0,
                                            train_generator=self.train_generator,
                                            valid_generator=self.valid_generator,
                                            optimizer=self.adam,
                                            nets=self.nets,
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V,
                                            contrastive_weights = self.contrastive_w,
                                            u_pts= self.u_pts,
                                            mono_phi_coef = self.mono_phi_coef,
                                            steep_step = self.steep_step,
                                            FB_coef = self.FB_coef,
                                            indx_init_SB=self.indx_init_SB,
                                            above_phi_c_coef = above_phi_c_coef,
                                            force_V_min_coef = force_V_min_coef,
                                            force_V0_coef = force_V0_coef,
                                            order_phi_H_coef = order_phi_H_coef,
                                            phim=self.phim,
                                        )
        
        self.solver.metrics_history['r2_loss'] = master_dict['loss']
        self.solver.metrics_history['train_loss'] = master_dict['train_loss']

        if 'add_loss' in master_dict:  # Check if it exists for backward compatibility
            self.solver.metrics_history['add_loss'] = master_dict['add_loss']
        if 'monotonic_phi_add_loss' in master_dict: 
            self.solver.metrics_history['monotonic_phi_add_loss'] = master_dict['monotonic_phi_add_loss']
        if 'FB_addloss' in master_dict: 
            self.solver.metrics_history['FB_addloss'] = master_dict['FB_addloss'] 
        if 'above_phi_c_addloss' in master_dict: 
            self.solver.metrics_history['above_phi_c_addloss'] = master_dict['above_phi_c_addloss']  
        if 'V_min_addloss' in master_dict:  
            self.solver.metrics_history['V_min_addloss'] = master_dict['V_min_addloss']  
        if 'DV_DDV_min_addloss' in master_dict: 
            self.solver.metrics_history['DV_DDV_min_addloss'] = master_dict['DV_DDV_min_addloss']  


        if 'DV_min_addloss' in master_dict: 
            self.solver.metrics_history['DV_min_addloss'] = master_dict['DV_min_addloss']  
        if 'DDV_min_addloss' in master_dict: 
            self.solver.metrics_history['DDV_min_addloss'] = master_dict['DDV_min_addloss']  


        if 'V0_addloss' in master_dict:  
            self.solver.metrics_history['V0_addloss'] = master_dict['V0_addloss']
        if 'phiH_ordering_addloss' in master_dict:  
            self.solver.metrics_history['phiH_ordering_addloss'] = master_dict['phiH_ordering_addloss']  

        self.solver.best_nets = np.ones_like(self.solver.nets)
        
        for i in range(len(self.solver.nets)):
            # Remove the .NN attribute - load directly to the network
            self.solver.nets[i].load_state_dict(master_dict['state_dict_solver'][i])
            # self.solver.nets[i].dense.load_state_dict(master_dict['state_dense_nets'][i])
            # self.solver.nets[i].alpha = master_dict['state_nets_alpha_param'][i]

            if master_dict['state_best_nets'][i] != None:
                print('best nets not None')
                self.solver.best_nets[i] = copy.deepcopy(self.solver.nets[i])
                self.solver.best_nets[i].load_state_dict(master_dict['state_best_nets'][i])
                # self.solver.best_nets[i].dense.load_state_dict(master_dict['state_dense_nets'][i])
                # self.solver.best_nets[i].alpha = master_dict['state_nets_alpha_param'][i]

                
        print(self.update_optimizer())
        print('Model succesfully loaded')
