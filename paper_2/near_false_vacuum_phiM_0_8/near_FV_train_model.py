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

from NNholo_near_FV import *
print('NNHolo initialized')




# Saving directory

directory = "FV_results"   # or a new directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
path_results = os.path.join(parent_dir, directory)

if os.path.exists(path_results) == False:
    os.makedirs(path_results, exist_ok=True)
    print("New directory created")



# Choose the phiM parameter value of the V(phi) potential
phim_param = [0.8] # recall near-false-vacuum range: [0.58, 0.8]


# Specify size of the NNs

solver_nets = [128,128,128,128]
V_nets = [64,64,64,64,64]
u_pts = 128


# Epochs for training

epochs = 150000



# Load the data
sofT_list = []
for k in range(len(phim_param)):

    phim_str = str(phim_param[k])
    phim_str = phim_str.replace('.', '_')

    data_path = "./Data/SofT/SofTphiM" + phim_str + ".txt"

    sofT_list.append(data_path)




# Initialize the NNholo class for training

c = NNholo(data_path = sofT_list[0], contrastive_weights=None, nets_loc_var=3, \
           saving_path = path_results, init_pt_curve=8, step=1, n_points=70, T_min=0.001, \
           add_index = True, solver_nets = solver_nets, V_nets = V_nets, \
           phim = phim_param[0], force_V_min_coef=0.01, force_DV_DDV_min_coef=1e-6, force_V0_coef=0.0)


# Optionally restart from a previously saved checkpoint instead of training from scratch
# c.load_results(path_results + "/model_epochs_" + str(epochs))   # resume at the epoch you stopped at
# c.load_results(f'{path_results}/trained_NN_more_epochs_')
# print('Net loaded')


c.sofT_curve()
plt.close()

store_mse = Store_MSE_Loss()

scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.985)
scheduler_cb = DoSchedulerStep(scheduler=scheduler)

potential_cb = BestValidationCallback()

c.update_optimizer()



# Train

c.solver.fit(max_epochs=epochs,
             callbacks=[potential_cb, scheduler_cb],
             tqdm_file=None)


# Optional alternative: train in chunks and reset the LR if it decays too far

# epochs_per_check = 50000   # check LR every 50k epochs
# total_epochs = epochs
# epochs_done = 0
#
# while epochs_done < total_epochs:
#     chunk_size = min(epochs_per_check, total_epochs - epochs_done)
#     c.solver.fit(max_epochs=chunk_size,
#                  callbacks=[potential_cb, scheduler_cb],
#                  tqdm_file=None)
#     epochs_done += chunk_size
#
#     current_lr = c.adam.param_groups[0]['lr']
#     if current_lr < 1e-5:
#         for g in c.adam.param_groups:
#             g['lr'] = 1e-4
#         print(f'LR was {current_lr:.2e}, reset to 1e-4 at epoch {epochs_done}')
#     else:
#         print(f'Epoch {epochs_done}: LR = {current_lr:.2e}')



# Save the results, metrics and figures

c.save_results(f'{path_results}/trained_NN_more')   # Save everything (weights, loss metrics, optimizer)

c.plot_loss()
plt.close()
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



elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas')