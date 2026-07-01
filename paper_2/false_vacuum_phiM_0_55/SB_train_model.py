import os
import time
t = time.process_time()
os.environ["DEV"] = "1"
os.environ["NEURODIFF_API_URL"] = "http://dev.neurodiff.io"
os.environ["NEURODIFF_API_KEY"] = 'tNaaIvvvdg72-c8VcTZRgpALsl0ns77ljEvxul6tG0E'
import warnings
warnings.filterwarnings("ignore")
from platform import python_version
import io
import pickle
import numpy
from tqdm.auto import tqdm
import random


from SB_main_pipeline import *






# Saving directory

directory = "SB_results"   # or a new directory e.g. "SB_results_from_scratch"
parent_dir = os.path.dirname(os.path.abspath(__file__))
path_results = os.path.join(parent_dir, directory) 

if os.path.exists(path_results) == False:
    os.makedirs(path_results) 
    print("New directory created")



# Choose the phiM parameter value of the V(phi) potential, this will be used for importing the data file for training the PINN model
phim_param=[0.55] 


# Specify size of the NNs for this phase of training: V(phi) from the data of SB with frozen pretrained V(phi) from FB data attached

solver_nets = [128,128,128,128]
V_nets = [64,64,64,64,64]
u_pts = 128


# Matching order of the derivatives of the pretrained FB potential (frozen weights) at phi_fv to the reparametrized full V(phi) potential. 
order = 3



# Epochs for training, and amount of repetitions (reps) of "train -> save -> load -> train -> save -> load -> ... (if necessary)"
check_epochs = 10
epochs = 10_000
reps = 1


# Epoch number at which to restart the training from the pretrained SB model (if necessary). 
# If None, then the training will start from scratch (and the initial load_path is that of the FB pretrained model).

loop_index = 23828600  # None



lr_vals = [1.5e-5, 1e-5]



# Load the data
sofT_list = []
for k in range(len(phim_param)):
        
    phim_str = str(phim_param[k])
    phim_str = phim_str.replace('.', '_')
    abs_path = os.path.dirname(os.path.abspath(__file__))


    data_path =  f"{parent_dir}/Data/Potentials_phim/SofTphiM" + phim_str + ".txt" 

    sofT_list.append(data_path)




# Path for the weights of pretrained FB NN model (will be frozen at this stage of training). 
# Such a model can be obtained by training the FB model from scratch with the FB_training.ipynb notebook, or by loading the pretrained FB model file upon request the authors.
first_branch_load_path = f'{parent_dir}/FB_results/NN_first_branch_epochs_1968656'



epochs_cumulative = 1_400_000
for j in range(reps):

    if loop_index is not None:
        second_branch_load_path = f'{path_results}/NN_second_branch_epochs_{loop_index}'
    else:
        second_branch_load_path = first_branch_load_path
        loop_index = 0





    # Initialize the NNholo class for training

    c = NNholo(data_path = sofT_list[0], load_path=second_branch_load_path, new_FB_path=first_branch_load_path, \
        contrastive_weights=None, nets_loc_var = 3, \
        saving_path = path_results, init_pt_curve=541, end_pt_curve=None, step=1, n_points=70, T_min=0.001, \
        add_index = True, solver_nets = solver_nets, V_nets=V_nets, u_pts = u_pts, phim=phim_param[0], \
        mono_phi_coef=0.0, first_branch_coef=0.0,  force_V0_coef=0, steep_step=1, \
        above_phi_c_coef = 1.0, order_phi_H_coef=0.0, \
        order_match_V_derivative = order, \
        force_V_min_coef = 0.01, force_DV_min_coef=1e-6, force_DDV_min_coef=1e-6, \
        seed=None)
    
    
    print('NNHolo initialized')
                
    store_mse = Store_MSE_Loss()

    scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.985)
    scheduler_cb = DoSchedulerStep(scheduler=scheduler)

    potential_cb = BestValidationCallback()



    ## Comment everything below if you just want to load the pretrained model, and visualize the results in the .ipynb notebook.
    ## v ------------------------------ v -------------------------------- v ------------------------- v ------------------------ v
    

    # Update the optimizer lr if necessary when sumarizing training
    if j%4 == 0: #and j!=0:
        lr = lr_vals[0]
        c.update_optimizer()
    else:
        lr = lr_vals[1]
        c.update_optimizer()
        

    # Train 

    if j==0:
        c.solver.fit(max_epochs=check_epochs, 
                    callbacks=[potential_cb, scheduler_cb], 
                    tqdm_file= tqdm(total=check_epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
        
        loop_index+=check_epochs
        epochs_cumulative+=check_epochs

    else:
        c.solver.fit(max_epochs=epochs, 
            callbacks=[potential_cb, scheduler_cb], 
            tqdm_file= None) # tqdm(total=epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
        
        loop_index+=epochs
        epochs_cumulative+=epochs


            

    # Save the results, metrics and figures

    c.save_results(f'{path_results}/NN_second_branch')   # Save everyhing (weights, loss metrics, optimizer) (recommended)
    # c.save_results_light(f'{path_results}/NN_second_branch')  # Save only weights and optimizer (less memory intensive, mainly for evaluation purposes, not for continue training afterwards)

    c.sofT_curve()
    plt.close()
    c.plot_loss(save_fig=True)
    plt.close()
    c.plot_loss(save_fig=True, past_epochs=10000+epochs_cumulative)
    plt.close()
    c.plot_potential(phim = phim_param[0], save_fig=False)
    plt.close()
    c.plot_potential_new(phim = phim_param[0], save_fig=True)
    plt.close()
    c.plot_residuals_in_u_color(save_fig=True)
    plt.close()
    c.plot_colored_sofT(save_fig=True, colormap='rainbow')
    plt.close()
    c.plot_solutions(step=5, save_fig = True)
    plt.close()
    c.compute_relative_error(n_pts = 1000, phim_param = phim_param[0], save_fig = True)
    plt.close()
    c.plot_separate_losses_new(past_epochs = None, step = 10000, save_fig=True)   # Cusotmize the step parameter to avoid too many points in the plot (for better visualization)
    plt.close()


    c.update_optimizer() # check LR at the end of the training interation


##################################



elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas' )
