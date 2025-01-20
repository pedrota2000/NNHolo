#%%
import os
import time
t = time.process_time()
os.environ["DEV"] = "1"
os.environ["NEURODIFF_API_URL"] = "http://dev.neurodiff.io"
os.environ["NEURODIFF_API_KEY"] = 'tNaaIvvvdg72-c8VcTZRgpALsl0ns77ljEvxul6tG0E'
import warnings
warnings.filterwarnings("ignore")
import dill
print(dill.__version__)
from platform import python_version

import io
import pickle
import shutil
#%%

import numpy
from master.NNholo_new_arch import *
print('NNHolo initialized')
from tqdm.auto import tqdm

# Directory 
directory = "Test"
# Parent Directory path 
parent_dir = "./Results/example/Architecture test/Comparison"
# Path 
path_results = os.path.join(parent_dir, directory) 

# # Directory
# directory = "test_1"
# # Parent Directory path 
# parent_dir = os.path.dirname(os.path.abspath(__file__))+"/phiM1_x8_neurons_solver_net"
# # Path 
if os.path.exists(path_results):
    shutil.rmtree(path_results)
os.mkdir(path_results) 
print("Directory '% s' created" % directory) 

# path_results = os.path.join(parent_dir, directory) 
# if os.path.exists(path_results) == False:
#     os.mkdir(path_results) 
#     print("Directory '% s' created" % directory)

c = NNholo(data_path = "./Data/SofT/SofTphiM5_0.txt", \
           saving_path = path_results,sampling_method='chebyshev2', init_pt_curve = 55, delta = 0.0, curriculum = 1.0)


# c = NNholo(data_path = os.path.dirname(os.path.abspath(__file__))+"/Data/SofT/Curve_sofT_case3of5.csv", \
#            saving_path = path_results)

#c.load_results(os.path.dirname(os.path.abspath(__file__))+"/trained_NN_test")

#print('Net loaded')

store_mse = Store_MSE_Loss()

scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.985)
scheduler_cb = DoSchedulerStep(scheduler=scheduler)

potential_cb = BestValidationCallback()

epochs = 100
c.solver.fit(max_epochs=epochs, 
             callbacks=[potential_cb, scheduler_cb], 
             tqdm_file=None)

#print(c.solver.hits)

#c.save_results(f'{path_results}/trained_NN_more_epochs')

c.plot_loss()
plt.close()
c.plot_potential(phim = 1.00)
plt.close()

c.save_results(f'{path_results}/trained_NN_more_epochs')

#do some stuff
elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas' )
#c.render()
# %%
