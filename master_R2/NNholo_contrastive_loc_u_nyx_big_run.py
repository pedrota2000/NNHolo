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
from NNholo_contrastive_loc_nets_w_seed_2_FIX import *
print('NNHolo initialized')
from tqdm.auto import tqdm

#NEW directory
directory = 'phiM_07'
# Parent Directory path 
parent_dir = os.path.dirname(os.path.abspath(__file__))+"/Results_R3/"
# Path 
path_results = os.path.join(parent_dir, directory) 

if os.path.exists(path_results) == False:
    os.makedirs(path_results) 
    print("New directory created")


phim_param=[0.7] #,0.8,0.75]#,0.72, 0.7,0.69, 0.68, 0.67, 0.66, 0.65, 0.645, 0.64, 0.635, 0.63, 0.625, 0.62, 0.615, 0.61, 0.605, 0.6]  #, 0.6, 0.58]

#solver_nets = [64,64,64]
#V_nets = [32,32,32,32]
#u_pts = 48

solver_nets = [128,128,128,128]
V_nets = [64,64,64,64,64]
u_pts = 128

epochs = 500_000
reps = 1

loop_index = 1_000_000
lr_vals = [1.5e-5, 1e-5]

sofT_list = []
for k in range(len(phim_param)):
        
    phim_str = str(phim_param[k])
    # Replace '.' with '_'s
    phim_str = phim_str.replace('.', '_')
    abs_path = os.path.dirname(os.path.abspath(__file__))
    # Get data that is one level up from the absolute path
    data_path = os.path.join(abs_path, 'Data', 'SofT', 'SofTphiM' + phim_str + '.txt')
    sofT_list.append(data_path)

contrastive_weights_path = parent_dir+'weights_per_eq_and_sofT_norm_along_eqs_0_7_pretrained_NN.npy'

for j in range(reps):
    c = NNholo(data_path = sofT_list[0], contrastive_weights=contrastive_weights_path, nets_loc_var=0, \
               saving_path = path_results, init_pt_curve=8, step=1, n_points=70, T_min=0.001, \
               add_index = True, solver_nets = solver_nets, V_nets = V_nets) #,sampling_method='chebyshev2')

    c.load_results(f'{path_results}/NN_contrastive_w_loc_u_0_7_seed_2_epochs_{loop_index}')

    store_mse = Store_MSE_Loss()

    scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.99)
    scheduler_cb = DoSchedulerStep(scheduler=scheduler)

    potential_cb = BestValidationCallback()

    # if j<=3:
    #     lr = lr_vals[0]
    # else:
    #     lr = lr_vals[1]
        
    # c.update_optimizer(lr)

    c.solver.fit(max_epochs=epochs, 
                 callbacks=[potential_cb, scheduler_cb], 
                 tqdm_file= None) # tqdm(total=epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
        

    c.save_results(f'{path_results}/NN_contrastive_w_loc_u_0_7_seed_2')
    #c.save_results(f'{path_results}/trained_NN_more_epochs')
    c.sofT_curve()
    plt.close()
    c.plot_loss(save_fig=True)
    plt.close()
    #c.plot_separate_losses(save_fig=True)
    #plt.close()
    c.plot_potential(phim = phim_param[0], save_fig=True)
    plt.close()
    c.plot_residuals_in_u_color(save_fig=True)
    plt.close()
    c.plot_colored_sofT(save_fig=True, colormap='rainbow')
    plt.close()
    c.plot_solutions(save_fig =True)
    plt.close()
    c.compute_relative_error(n_pts = 1000, phim_param = phim_param[0], save_fig = True)
    plt.close()
    ###############################

    c.update_optimizer()

    loop_index+=epochs

##################################

elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas' )
