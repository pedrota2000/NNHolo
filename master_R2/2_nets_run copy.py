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
from NNholo_contrastive_loc_nets_w_seed_2 import *
print('NNHolo initialized')
from tqdm.auto import tqdm

from NNholo_contrastive_loc_nets_w_seed_2 import *
from tqdm.auto import tqdm


plt.rcParams['font.family'] = 'DejaVu Serif'  # This is available by default with matplotlib
warnings.filterwarnings("ignore", module="matplotlib.font_manager")


#NEW directory
directory = 'Cluster addloss -log_sigmoid+10 cosine_LR'
# Parent Directory path 
parent_dir = os.path.dirname(os.path.abspath(__file__))+"/Cluster runs/Round 3"
# Path 
path_results = os.path.join(parent_dir, directory) 

if os.path.exists(path_results) == False:
    os.mkdir(path_results) 
    print("New directory created")
else:
    print("Directory already exists")

#load_path = "/home/pedro/NNHolo_new_life/raw_runs/test_06/NN_MH_heist_epochs_100000"
phim_param=[0.8] #,0.8,0.75]#,0.72, 0.7,0.69, 0.68, 0.67, 0.66, 0.65, 0.645, 0.64, 0.635, 0.63, 0.625, 0.62, 0.615, 0.61, 0.605, 0.6]  #, 0.6, 0.58]

sofT_list = []
for k in range(len(phim_param)):
        
    phim_str = str(phim_param[k])
    # Replace '.' with '_'
    phim_str = phim_str.replace('.', '_')

    sofT_path = "Data/SofT/SofTphiM" + phim_str + ".txt"

    sofT_list.append(sofT_path)

contrastive_weights_path = f'{parent_dir}/weights_per_eq_and_interp_IR_sofT_norm_along_eqs_0_8_pau_base_run.npy'

# solver_nets = [64,64,64,64,64]
# V_nets = [64,64,64,64]
# u_pts = 128

epochs = 100000

sofT_list = []
for k in range(len(phim_param)):
        
    phim_str = str(phim_param[k])
    # Replace '.' with '_'
    phim_str = phim_str.replace('.', '_')

    sofT_path = os.path.dirname(os.path.abspath(__file__))+"/Data/SofT/SofTphiM" + phim_str + ".txt"

    sofT_list.append(sofT_path)
    

c = NNholo(data_path = sofT_list[0], contrastive_weights=contrastive_weights_path, nets_loc_var=0, \
           saving_path = path_results, init_pt_curve=8, step=1, n_points=70, T_min=0.001, \
           add_index = True, solver_nets = [64,64,64], V_nets = [32,32,32,32]) #,sampling_method='chebyshev2')

re_evol = [0.0]


#c.load_results(path_results + "/model_epochs_" + str(epochs+100000)) # The number of epochs at which you stopped. May be trained_NN_more__epochs

#print('Net loaded')

c.sofT_curve()
plt.close()
    
store_mse = Store_MSE_Loss()

#scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=2000, gamma=0.965)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(c.adam, T_max=100000)

scheduler_cb = DoSchedulerStep(scheduler=scheduler)

potential_cb = BestValidationCallback()

param_tracker = TrackParameters()

c.update_optimizer(0)

c.solver.fit(max_epochs=epochs, # Change max_epochs to something else when you re-train for the 2nd time
             callbacks=[potential_cb, scheduler_cb, param_tracker], #nets_param_tracker], 
             tqdm_file=tqdm(total=epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))

c.save_results(f'{path_results}/model')
#c.save_results(f'{path_results}/trained_NN_more')
c.plot_loss(addloss=True)
plt.close()
c.plot_loss(addloss=False)
plt.close()
#c.plot_separate_losses(save_fig=True)
#plt.close()
c.plot_potential(phim = phim_param[0], save_fig=True)
plt.close()
c.plot_residuals_in_u(max_bound = False, print_overbound = False, save_fig=True)
plt.close()
c.plot_colored_sofT(colormap = 'rainbow')
plt.close()
c.plot_residuals()
plt.close()
c.plot_solutions()
plt.close()
c.plot_3D_sofT()
plt.close()
#c.compare_to_yago(save_fig=True)
#c.save_results_new(path_results_now + "/trained_NN_phim_0_" + decimal_phim_now)

elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas' )
