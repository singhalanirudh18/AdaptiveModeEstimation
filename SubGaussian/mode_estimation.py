import os
import time
import numpy as np
from find_knn import FindKNN
import heapq


#Finds the Mode of the dataset Y' s.t. Y is a n*(n-1) table which stores the distances
def AdaptiveModeEstimation(Y,n,k,delta,which_beta,step_size,c_beta=0.75,sigma=0.25,epsilon=0.001,verbose=False) :
    
    knn_found=np.zeros(n)
    knn_estimate=np.zeros(n)
    UCBs = np.zeros(n)
    LCBs = np.zeros(n)
    LCBs_heap = []
    DS = np.zeros([n,n-1],dtype=np.float64)
    TS= np.zeros([n,n-1],dtype=np.uint64)
    
    count=0
    UCB_arr = np.zeros(n-1)
    LCB_arr = np.zeros(n-1)

    ##initialize all the points by running the Findk-NN algo once 
    for i in range(n) :
        DS_i,TS_i,b_i,knn_found_i=FindKNN(Y[i,:], k, delta, which_beta, c_beta,sigma).query(1, ds=DS[i,:], ts=TS[i,:])
        DS[i,:]=DS_i
        TS[i,:]=TS_i
        knn_estimate[i]=b_i
        if knn_found_i :
            knn_found[i]=1
        for j2 in range(n-1):
            UCB_arr[j2] = DS_i[j2] + beta_my(TS_i[j2], delta, n-1, c_beta)
            LCB_arr[j2] = DS_i[j2] - beta_my(TS_i[j2], delta, n-1, c_beta)
        UCBs[i] = np.partition(UCB_arr,k-1)[k-1] #select the kth highest UCB as the UCB of k-NN distance
        LCBs[i] = np.partition(LCB_arr,k-1)[k-1] #select the kth highest LCB as the UCB of k-NN distance
        heapq.heappush(LCBs_heap,(LCBs[i],i))

    
    l1 = LCBs_heap[0][1]
    l2 = second_smallest(LCBs_heap[1],LCBs_heap[2])

    UCB_arr = np.zeros(n-1)
    LCB_arr = np.zeros(n-1)
    count =0
    
    if verbose :
        print('{} : Current min UCB : {}, Current second min LCB : {}'.format(count, min_UCB_current,l2[0]))
    
    while(LCBs[l2]<=UCBs[l1]-epsilon) : #stopping criteria is not met, epsilon added for faster termination
        
        count+=1
        i = l1 ## point to sample
        
        ##call the inner-loop
        if knn_found[i]==0 :  #if knn is not identified
            DS_i,TS_i,b_i,knn_found_i=FindKNN(Y[i,:], k, delta, which_beta, c_beta,sigma).query(step_size, ds=DS[i,:], ts=TS[i,:])
            DS[i,:]=DS_i
            TS[i,:]=TS_i
            knn_estimate[i]=b_i
            if knn_found_i :
                knn_found[i]=1    
            for j2 in range(n-1):
                UCB_arr[j2] = DS_i[j2] + beta_my(TS_i[j2], delta, n-1, c_beta)
                LCB_arr[j2] = DS_i[j2] - beta_my(TS_i[j2], delta, n-1, c_beta)
            UCBs[i] = np.partition(UCB_arr,k-1)[k-1]
            LCBs[i] = np.partition(LCB_arr,k-1)[k-1]
        
        else : #if knn is identified
            b_i = int(knn_estimate[i])
            for step in range(int(step_size)) :
                TS[i,b_i] += 1
                DS[i,b_i] = (DS[i,b_i] * (TS[i,b_i] - 1)) / TS[i,b_i] + (np.random.normal(Y[i,b_i],sigma)) / TS[i,b_i]
            UCBs[i] = DS[i,b_i] + beta_my(TS[i,b_i], delta, n-1, c_beta)
            LCBs[i] = DS[i,b_i] - beta_my(TS[i,b_i], delta, n-1, c_beta)

        heapq.heapreplace(LCBs_heap,(LCBs[i],i))
        l1 = LCBs_heap[0][1]
        l2 = second_smallest(LCBs_heap[1],LCBs_heap[2])
        
        if ((count%100 ==0) and verbose) :
            print('{} : Current min UCB : {}, Current second min LCB : {}'.format(count, UCBs[l1]-epsilon,l2[0]))
        if (LCBs[l2]==UCBs[l1]) and (LCBs[l1]==UCBs[l1]) :
            break

    num_queries_final=np.sum(TS)
    return l1, num_queries_final

#to find l2
def second_smallest(child1,child2) :
    if child1[0]<child2[0] :
        return child1[1]
    else :
        return child2[1]

#To calculate the confidence intervals 
def beta_my(ts, delta, n, c_beta, m= 64*64*3):
    if c_beta == 0:
        log_reciprocal_delta = -np.log(delta/(n*(n-1)))
        beta = log_reciprocal_delta + 3 * np.log(log_reciprocal_delta) + 1.5 * np.log(1 + np.log(ts))
        ans = np.sqrt(2 * beta / ts)
    else :
        ans = np.sqrt(c_beta * np.log(1 + (1 + np.log(ts)) * (n*(n-1)) / delta) / ts)
    return ans

#finding the mode using the naive method
def ModeEstimationNaive(Y,n,k,epsilon=0.001) :
    mode_candidates = np.zeros(n)
    if (epsilon!=0.001) :
        print('Epsilon : {}'.format(epsilon))
    for j in range(n) :
        dists = np.linalg.norm(np.delete(Y,j,0) - Y[j,:], axis=1)
        idx = np.argsort(dists)
        mode_candidates[j]=dists[idx[k-1]]
        
    idx_final=np.argsort(mode_candidates)
    dist_sort = np.sort(mode_candidates)
    mode_set = np.array(idx_final[0])
    m=64*64*3
    d1=(dist_sort[0]**2)/m
    d2=(dist_sort[1]**2)/m
    for i in range(n-1):
        if(dist_sort[i+1]-dist_sort[0]<=epsilon): #if the problem is too hard select a new dataset
            return -1
            mode_set = np.append(mode_set,idx_final[i+1])
    return mode_set
