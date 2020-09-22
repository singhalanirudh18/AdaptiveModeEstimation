import os
import time
import numpy as np
from mode_estimation import *

tiny_imagenet_dataset = np.load('../Tiny_imagenet_dataset.npy')

def run_random_mode_estimation(m, n, k, delta, which_beta,step_size, c_beta=0.75, database= tiny_imagenet_dataset, verbose=False):
    while(1) : 
        #select n random points to run AME on
        Y = database[np.random.choice(database.shape[0], size=n, replace=False), :].copy()
        ## Find actual mode using naive method
        mode_set = ModeEstimationNaive(Y,n,k)
        if (mode_set!=-1) :
            break

    ## Adaptive Mode Estimation
    start_time=time.time()
    estimated_mode,num_queries = AdaptiveModeEstimation(Y,n,m,k,delta,which_beta,step_size,c_beta)
    total_time=time.time()-start_time
    
    accuracy=0
    if np.size(np.intersect1d(mode_set,estimated_mode)) >0 :
        accuracy=1
    return accuracy, total_time, num_queries

m = 64*64*3
n = 100
k = int(0.1*n)
delta = 1e-3
c_beta=0.03
step_size = m*n/100 #we run step_size calls of FindkNN in single iteration for faster implementation
        
which_beta = 'not theoretical'
if which_beta=='theoretical' :
    c_beta=0

N=10 #number of iterations

print('Starting')

cumm_acc = 0
cumm_time = 0
cumm_num_queries = 0.0

for i in range(N):
    curr_acc,curr_time,curr_num_queries=run_random_mode_estimation(m,n,k,delta,which_beta,step_size,c_beta,database=tiny_imagenet_dataset)
    cumm_acc+=curr_acc
    cumm_time+=curr_time
    cumm_num_queries+=float(curr_num_queries)/(m*n*n)
    print('{} : Accuracy : {} ; Time Taken : {}; Number of Queries : {}'.format(i+1,float(cumm_acc/(i+1)),cumm_time/(i+1),cumm_num_queries/(i+1)))
    