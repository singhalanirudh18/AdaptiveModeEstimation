import os
import time
import numpy as np
from mode_estimation import *

tiny_imagenet_dataset = np.load('../Tiny_imagenet_dataset.npy')

#runs AME after calculating distances
def run_random_mode_estimation(m, n, k, delta, which_beta,step_size, c_beta=0.75, sigma=0.25, epsilon=0.001,database= tiny_imagenet_dataset, verbose=False):
    while(1) : 
        Y = database[np.random.choice(database.shape[0], size=n, replace=False), :].copy() #select n points randomly to run AME on
        
        #find distances
        Y_distances = np.zeros([n,n-1])
        for i in range(n) :
            j2=0
            for j in range(n) :
                if (i!=j) :
                    Y_distances[i,j2] = (np.linalg.norm(Y[i,:]-Y[j,:],ord=2)**2)/m
                    j2+=1
        
        #Actual mode using Naive method
        mode_set = ModeEstimationNaive(Y,n,k,epsilon)
        
        if (mode_set!=-1) : #problem has acceptable level of hardness
            break

    ## Adaptive Mode Estimation
    start_time=time.time()
    estimated_mode,num_queries = AdaptiveModeEstimation(Y_distances,n,k,delta,which_beta,step_size,c_beta,sigma,epsilon)
    total_time=time.time()-start_time
    
    accuracy=0
    if np.size(np.intersect1d(mode_set,estimated_mode)) >0 :
        accuracy=1
    
    return accuracy, total_time, num_queries

m = 64*64*3
n = 100
k = int(0.1*n)
delta = 1e-3
c_beta=0.01
sigma=0.1 #noise level
epsilon = 0.001
step_size = n*n/10 #we run step_size calls of FindkNN in single iteration for faster implementation        
which_beta = 'not theoretical'

if which_beta=='theoretical' :
    c_beta=0

N = 10 #number of iterations

print('Starting')

cumm_acc = 0
cumm_time = 0
cumm_num_queries = 0

for i in range(N):
    curr_acc,curr_time,curr_num_queries=run_random_mode_estimation(m,n,k,delta,which_beta,step_size,c_beta,sigma,epsilon)
    cumm_acc+=curr_acc
    cumm_time+=curr_time
    cumm_num_queries+=curr_num_queries
    print('{} : Accuracy : {} ; Time Taken : {}; Number of Queries : {}'.format(i+1,float(cumm_acc/(i+1)),cumm_time/(i+1),int(cumm_num_queries/(i+1))))