import numpy as np

import random
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from scipy.linalg import eig, eigh
from scipy.stats import unitary_group

import matplotlib.pyplot as plt
import itertools as it
import sys
import progressbar
import time

import os
from pathlib import Path

from functions import *

from joblib import Parallel, delayed

## MAIN EXECUTION
if __name__ == "__main__":
    
    # obtain input from command line
    
    # input parameters
    N = int(sys.argv[1]) # number of modes
    p = float(sys.argv[2]) # squeezing
    N_sample = int(sys.argv[3]) # number of samples
    clicks_per_sample = int(sys.argv[4]) # clicks per sample
    
    print("Running for N="+str(N) + ", p="+str(p) + ", N_sample="+str(N_sample) + ", clicks_per_sample="+str(clicks_per_sample) )
    
    # where to save data
    try:
        folder = str(sys.argv[5]) 
        save = True
        print("Data will be saved in folder " + folder)
    except:
        save = False
        print("Data not saved")
        
    try:
        N_traj = int(sys.argv[6]) # number of trajectories in parallel, set to number of cores if not specified
    except:
        N_traj = os.cpu_count()
    
    # start state   
    G_start = np.diag([s%2 for s in range(N)])
        
    def get_instance():
        
            d_Np = sample_trajectory(N,
                             U_func=U_layered,
                             p=p, 
                             N_sample=N_sample, 
                             clicks_per_sample=clicks_per_sample, 
                             it_normalize=100, 
                             G_start=G_start, 
                             track=False)
            
            return np.array(d_Np["S"])
            
            
    # parallel function execution, evaluate run time
    print("Obtaining {} samples...".format(N_traj))
    t1 = time.time()
    S_dat = Parallel(n_jobs=-1)(delayed(get_instance)() for i in range(N_traj))
    t2 = time.time()
    
    print("Finished. Running time: " + str(t2-t1))
    
    # save data
    if save:
        filename = "/dat_N_" + str(N) + "_p_{:.2f}".format(p) + "_Nsample_{}".format(N_sample) + "_cps_{}".format(clicks_per_sample) 
        fullname = folder + filename + ".npy"
        
        # append
        if Path(fullname).is_file():
            S_prev = np.load(fullname)
            np.save(fullname, np.concatenate((S_prev,S_dat), axis=0))
        # create new file
        else:
            np.save( fullname, np.array(S_dat) )
    
                             
                             
                             
                             