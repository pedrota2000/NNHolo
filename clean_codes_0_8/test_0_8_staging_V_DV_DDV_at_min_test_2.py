import os
import time
t = time.process_time()
os.environ["DEV"] = "1"
os.environ["NEURODIFF_API_URL"] = "http://dev.neurodiff.io"
os.environ["NEURODIFF_API_KEY"] = 'tNaaIvvvdg72-c8VcTZRgpALsl0ns77ljEvxul6tG0E'
import warnings
warnings.filterwarnings("ignore")
#import dill
#print(dill.__version__)
from platform import python_version

import io
import pickle

import numpy
from rutine_0_8_staging_DV_DDV_min import *
print('NNHolo initialized')
from tqdm.auto import tqdm

#NEW directory #
import random


order = 3


directory = "staging_V_DV_DDV_at_min_test_2"
# Parent Directory path 
parent_dir = os.path.dirname(os.path.abspath(__file__))+"/phiM_0_8/"
# Path 
path_results = os.path.join(parent_dir, directory) 

if os.path.exists(path_results) == False:
    os.makedirs(path_results) 
    print("New directory created")


phim_param=[0.8] #,0.8,0.75]#,0.72, 0.7,0.69, 0.68, 0.67, 0.66, 0.65, 0.645, 0.64, 0.635, 0.63, 0.625, 0.62, 0.615, 0.61, 0.605, 0.6]  #, 0.6, 0.58]

#solver_nets = [64,64,64]
#V_nets = [32,32,32,32]
#u_pts = 48

solver_nets = [128,128,128,128]
V_nets = [64,64,64,64,64]
u_pts = 128

check_epochs = 100
epochs = 100_000
reps = 1

# epochs = 10
# reps = 2

loop_index = 0

lr_vals = [1.5e-5, 1e-5]

sofT_list = []
for k in range(len(phim_param)):
        
    phim_str = str(phim_param[k])
    phim_str = phim_str.replace('.', '_')
    abs_path = os.path.dirname(os.path.abspath(__file__))
    # Get data that is one level up from the absolute path
    # data_path = os.path.join(abs_path, 'Data', 'SofT', 'SofTphiM' + phim_str + '.txt')

    # data_path = "/Users/pablo/Desktop/NNholo/MH_flattening_paper_3_to_1/Data/Potentials_phim/SofTphiM" + phim_str + ".txt" # Local path
    data_path =  "/home/ptejerina/NNholo/Data/Potentials_phim/SofTphiM" + phim_str + ".txt"   # NYX path

    sofT_list.append(data_path)

contrastive_weights_path = None #parent_dir+'weights_per_eq_and_sofT_norm_along_eqs_0_6_pretrained_NN.npy'


c = NNholo(data_path = sofT_list[0], contrastive_weights=None, nets_loc_var = 3, \
    saving_path = path_results, init_pt_curve=8, end_pt_curve=None, step=1, n_points=70, T_min=0.001, \
    add_index = True, solver_nets = solver_nets, V_nets=V_nets, u_pts = u_pts, \
    force_V_min_coef = 1e-2, force_DV_DDV_min_coef = 1e-3, force_V0_coef=0, staging_addloss_epoch = 100_000, \
    seed = 'random')
    
store_mse = Store_MSE_Loss()

scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.985)
scheduler_cb = DoSchedulerStep(scheduler=scheduler)

potential_cb = BestValidationCallback()


for j in range(reps):

    # if j%2 != 0:
    #     lr = lr_vals[0]
    #     c.update_optimizer(7e-7)

    # else:
    #     lr = lr_vals[1]
    #     c.update_optimizer()
        
    if j==0:
        c.solver.fit(max_epochs=check_epochs, 
                    callbacks=[potential_cb, scheduler_cb], 
                    tqdm_file= tqdm(total=check_epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
    else:
        c.solver.fit(max_epochs=epochs, 
            callbacks=[potential_cb, scheduler_cb], 
            tqdm_file= None) # tqdm(total=epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
        

    c.save_results(f'{path_results}/NN_0_8')
    #c.save_results(f'{path_results}/trained_NN_more_epochs')
    c.sofT_curve()
    plt.close()
    c.plot_loss(save_fig=True)
    plt.close()
    c.plot_potential(phim = phim_param[0], save_fig=True)
    plt.close()
    # c.plot_potential_new(phim = phim_param[0], save_fig=True)
    plt.close()
    c.plot_residuals_in_u_color(save_fig=True)
    plt.close()
    c.plot_colored_sofT(save_fig=True, colormap='rainbow')
    plt.close()
    c.plot_solutions(save_fig =True)
    plt.close()
    c.compute_relative_error(n_pts = 1000, phim_param = phim_param[0], save_fig = True)
    plt.close()
    c.plot_separate_losses_new(past_epochs = int(2*(j+1)*epochs), save_fig=True)
    plt.close()

    ###############################

    c.update_optimizer()

    loop_index+=epochs

##################################

elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas' )
