import os
import time
t = time.process_time()
os.environ["DEV"] = "1"
os.environ["NEURODIFF_API_URL"] = "http://dev.neurodiff.io"
os.environ["NEURODIFF_API_KEY"] = 'tNaaIvvvdg72-c8VcTZRgpALsl0ns77ljEvxul6tG0E'
import warnings
warnings.filterwarnings("ignore")
import dill
dill.__version__
from platform import python_version

#print(python_version())
class train_model():
    def __init__(self,data_path,results_path = "train_results",sampling = "chebyshev2"):
        import numpy
        from NNholo import NNholo
        from tqdm.auto import tqdm
        # Directory
        directory = results_path
        # Parent Directory path 
        #parent_dir = os.path.dirname(os.path.abspath(__file__))+"/phiM1"
        # Path 
        path_results = os.path.join(directory) 
        if os.path.exists(path_results) == False:
            os.mkdir(path_results) 
            print("Directory '% s' created" % directory)
        c = NNholo(data_path = os.path.dirname(data_path,saving_path = path_results,sampling_method=sampling)
        
        
        store_mse = Store_MSE_Loss()
        
        scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.985)
        scheduler_cb = DoSchedulerStep(scheduler=scheduler)
        
        potential_cb = BestValidationCallback()
        
        
        epochs = Input("Number of epochs to train:" )
        c.solver.fit(max_epochs=epochs, 
                     callbacks=[potential_cb, scheduler_cb])
        
        print(c.solver.hits)
        
        c.save_results(f'{path_results}/trained_NN_test')
        
        c.plot_loss()
        c.plot_potential(potential_cb,phim = 1.0)
        
        #do some stuff
        elapsed_time = time.process_time() - t
        print(str(elapsed_time/3600) + ' horas' )
