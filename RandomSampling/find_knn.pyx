cimport cython

from libc.math cimport log, sqrt, pow
from libc.stdlib cimport rand, RAND_MAX

from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from heap cimport Heap

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
UTYPE = np.uint64
ctypedef np.uint64_t UTYPE_t

ctypedef DTYPE_t (*CONFIDENCE_BOUND_FUN_t)(UTYPE_t, DTYPE_t, UTYPE_t, DTYPE_t)

cdef inline DTYPE_t beta_alternative(UTYPE_t u, DTYPE_t delta, UTYPE_t n, DTYPE_t beta_c):
    return sqrt(beta_c * log(1 + (1 + log(u)) * (n*(n-1)) / delta) / u)

# beta_c is unused in the theoretical bound
cdef inline DTYPE_t beta_theoretical(UTYPE_t u, DTYPE_t delta, UTYPE_t n, DTYPE_t beta_c):
    return sqrt(2 * beta(u, delta_prime(delta, n*(n-1))) / u)

cdef inline DTYPE_t delta_prime(DTYPE_t delta, UTYPE_t n):
    return 1 - pow(1 - delta, 1.0 / n)

cdef inline DTYPE_t beta(DTYPE_t u, DTYPE_t delta):
    cdef DTYPE_t log_reciprocal_delta = -log(delta)
    return log_reciprocal_delta + 3 * log(log_reciprocal_delta) + 1.5 * log(1 + log(u))

cpdef inline UTYPE_t rand_upto(UTYPE_t n):
    return <UTYPE_t> (<DTYPE_t> rand() / (<DTYPE_t> RAND_MAX + 1.0) * n)

cdef inline bool lte(pair[DTYPE_t, UTYPE_t] a, pair[DTYPE_t, UTYPE_t] b):
    return a.first <= b.first

@cython.boundscheck(False)
@cython.wraparound(False)


cdef find_knn(DTYPE_t[:] x, UTYPE_t num_iter, DTYPE_t[:, ::1] Y, UTYPE_t k, DTYPE_t delta, str which_beta, DTYPE_t beta_c, DTYPE_t[:] ds_input, UTYPE_t[:] ts_input):
    
    cdef UTYPE_t n = Y.shape[0]
    cdef UTYPE_t m = Y.shape[1]

    cdef CONFIDENCE_BOUND_FUN_t beta
    if which_beta == 'theoretical':
        beta = beta_theoretical
    else:
        beta = beta_alternative


    if k > n:
        raise ValueError('Requesting too many neighbors (k > n).')

    # initialize the distance estimates
   
    cdef DTYPE_t[:] ds = ds_input
    cdef UTYPE_t[:] ts = ts_input
    
    cdef UTYPE_t i, j
    for i in range(n):
        if(ts[i] == 0):
            j = rand_upto(m)
            ds[i] = (x[j] - Y[i, j]) ** 2
            ts[i] = 1
    
    # sort into heaps, as follows:
    
    cdef UTYPE_t[:] ds_sort_idx = np.argsort(ds).astype(UTYPE)

    ## first k-1 elements

    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_top_v = vector[pair[DTYPE_t, UTYPE_t]](k-1)
    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_top_dist_v = vector[pair[DTYPE_t, UTYPE_t]](k-1)
    for i in range(k-1):
        j = ds_sort_idx[i]
        heap_top_v[i] = pair[DTYPE_t, UTYPE_t](ds[j] + beta(ts[j], delta, n, beta_c), j)
        heap_top_dist_v[i] = pair[DTYPE_t, UTYPE_t](ds[j], j)
    cdef Heap heap_top = Heap(heap_top_v, is_max_heap=True)
    cdef Heap heap_top_dist = Heap(heap_top_dist_v, is_max_heap=True)
    
    ## kth element
 
    j = ds_sort_idx[k-1]
    cdef DTYPE_t[:] k_element_parametrs = np.zeros(3)
    #cdef vector[DTYPE_t](3) k_element_parametrs
    k_element_parametrs[0] = ds[j]
    k_element_parametrs[1] = ds[j] + beta(ts[j], delta, n, beta_c) #stores the UCB
    k_element_parametrs[2] = ds[j] - beta(ts[j], delta, n, beta_c) #stores the LCB
    cdef pair[DTYPE_t[:],UTYPE_t] b = pair[DTYPE_t[:],UTYPE_t] (k_element_parametrs,j) #stores the k^th element, its UCB and its LCB


    #stores the next n-k elements

    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_bottom_v = vector[pair[DTYPE_t, UTYPE_t]](n - k)
    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_bottom_dist_v = vector[pair[DTYPE_t, UTYPE_t]](n - k)
    for i in range(n - k):
        j = ds_sort_idx[i + k]
        heap_bottom_v[i] = pair[DTYPE_t, UTYPE_t](ds[j] - beta(ts[j], delta, n, beta_c), j)
        heap_bottom_dist_v[i] = pair[DTYPE_t, UTYPE_t](ds[j], j)
    cdef Heap heap_bottom = Heap(heap_bottom_v, is_max_heap=False)
    cdef Heap heap_bottom_dist = Heap(heap_bottom_dist_v, is_max_heap=False)
    
    # main loop

    cdef UTYPE_t t
    cdef pair[DTYPE_t, UTYPE_t] a1, a2
    cdef DTYPE_t beta_i, beta_j
    cdef bool heaps_sorted, knn_found
    knn_found = False
    for t in range(num_iter):
        
        knn_found = True
        a1 = heap_top.head()
        a2 = heap_bottom.head()

        #update the point with highest UCB in first k-1 points if it is not eliminated

        if a1.first >= b.first[2] : #UCB[a1]>=LCB[b]
            knn_found = False
            i = a1.second
            beta_i = update_estimate_get_confidence(x, Y, ds, ts, i, delta, beta, beta_c)
            heap_top.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i] + beta_i, i))
            heap_top_dist.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i], i))   

        # update the point with lowest LCB in bottom n-k if conditionis met
        if a2.first <= b.first[1] : #LCB[a2]<= UCB[b]
            knn_found = False
            i = a2.second
            beta_i = update_estimate_get_confidence(x, Y, ds, ts, i, delta, beta, beta_c)
            heap_bottom.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i] - beta_i, i))
            heap_bottom_dist.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i], i))

        #update m 
        i = b.second
        beta_i = update_estimate_get_confidence(x, Y, ds, ts, i, delta, beta, beta_c)
        b.first[0] = ds[i] #distance
        b.first[1] = ds[i]+beta_i #UCB
        b.first[2] = ds[i]-beta_i #LCB

        if knn_found :
            continue # knn found terminate

        # sort heaps

        heaps_sorted = False

        while not heaps_sorted:
            
            heaps_sorted = True

            a1 = heap_top_dist.head()
            
            if a1.first > b.first[0]:

                heaps_sorted = False
                i = a1.second
                j = b.second

                a2 = heap_top.get_by_key(i)

                beta_i = a2.first - ds[i]
                beta_j = b.first[1] - b.first[0]

                heap_top_dist.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[j], j))
                heap_top.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[j] + beta_j, j))
                
                b.second = i
                b.first[0] = ds[i]
                b.first[1] = ds[i]+beta_i
                b.first[2] = ds[i]-beta_i 
            
            a1 = heap_bottom_dist.head()

            if a1.first < b.first[0]:

                heaps_sorted = False
                i = a1.second
                j = b.second

                a2 = heap_bottom.get_by_key(i)

                beta_i = ds[i] - a2.first
                beta_j = b.first[1] - b.first[0]

                heap_bottom_dist.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[j], j))
                heap_bottom.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[j] - beta_j, j))
                
                b.second = i
                b.first[0] = ds[i]
                b.first[1] = ds[i]+beta_i
                b.first[2] = ds[i]-beta_i

    
    return ds,ts, b.second, knn_found
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t update_estimate_get_confidence(DTYPE_t[:] x, DTYPE_t[:, ::1] Y, DTYPE_t[:] ds, UTYPE_t[:] ts, UTYPE_t i, DTYPE_t delta, CONFIDENCE_BOUND_FUN_t beta, DTYPE_t beta_c):
    
    cdef UTYPE_t n = Y.shape[0]
    cdef UTYPE_t m = Y.shape[1]

    if ts[i] < m:

        ts[i] += 1

        if ts[i] == m:
            ts[i] = 2*m
            ds[i] = distance2(x, Y, i) / m
            return 0

        else:

            j = rand_upto(m)
            ds[i] = (ds[i] * (ts[i] - 1)) / ts[i] + ((x[j] - Y[i, j]) ** 2) / ts[i]
            return beta(ts[i], delta, n, beta_c)

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t distance2(DTYPE_t[:] x, DTYPE_t[:, ::1] Y, UTYPE_t i):

    cdef DTYPE_t s = 0
    cdef UTYPE_t j
    for j in range(Y.shape[1]):
        s += (x[j] - Y[i, j]) ** 2

    return s


class FindKNN(object):

    def __init__(self, Y, k, delta=1e-6, which_beta='theoretical', beta_c=0.75):

        self.Y = Y
        self.k = k
        self.delta = delta
        self.which_beta = which_beta
        self.beta_c = beta_c

    def query(self, x, num_iter, ds= None, ts= None):
        ds_out,ts_out, b, knn_found = find_knn(x,num_iter, self.Y, self.k, self.delta, self.which_beta, self.beta_c, ds, ts)
        return ds_out,ts_out, b, knn_found
