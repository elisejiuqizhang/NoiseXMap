import numpy as np
from .DelayEmd import time_delay_embed
from .CM_simplex import CM_rep_simplex # get_distance, get_nearest_distances

def fNN(ts, tau=1, emd_min=2, emd_max=24, emd_step=2, knn=10, ratio_thres=15, A_tol=2, L=None, false_tol=0.05):
    """ False nearest neighbors (fNN) algorithm for a univariate time series 
    (Kennel’s Algorithm as described in https://www.hindawi.com/journals/jcs/2015/932750/#).
    Args:
        ts         time series (univariate)
        tau        time delay
        emd_min    minimum embedding dimension
        emd_max    maximum embedding dimension
        emd_step   step size for embedding dimension
        knn        neighborhood size
        ratio_thres  threshold for fNN, if the nearest neighbor distance ratio increases as the dim increases, then it is a false nearest neighbor.
        A_tol      second threshold
        L          max length/time length, if None then full length
        false_tol  tolerance for percentage of points having false nearest neighbors"""
    
    dists_info = {}
    nn_info = {}
    fNN_info = {}
    for emd in range(emd_min, emd_max+1, emd_step):
        # get distance matrix of the current embedding dimension
        DE= time_delay_embed(ts, tau, emd)
        if L is not None:
            DE = DE[:L]
        dists=CM_rep_simplex.get_distance_vanilla(DE)
        dists_info[emd]=dists # store the distance matrix
        
        # get nearest neighbor distance of each point at the current embedding dimension
        nn_info[emd]=[] # list to store (nearest_ids, nearest_dists)
        for t_idx in range(DE.shape[0]):
            nearest_ids, nearest_dists = CM_rep_simplex.get_nearest_distances(dists, t_idx, knn)
            nn_info[emd].append((nearest_ids, nearest_dists))

        # calculate the nearest neighbor distance ratio for each embedding dimension
        if emd>emd_min:
            fNN_info[emd]=[]
            for t_idx in range(DE.shape[0]): # for each time points
                # nearest indices and distances of the previous embedding dimension
                prev_nearest_ids, prev_nearest_dists = nn_info[emd-emd_step][t_idx]

                # using previous indices, retrieve distances of the current embedding dimension
                currDists_prevIds = dists_info[emd][t_idx, prev_nearest_ids]

                # calculate the nearest neighbor distance ratio: current / previous
                ratio = np.sqrt(np.abs(currDists_prevIds**2-prev_nearest_dists**2)/(prev_nearest_dists**2+1e-8))
                ratio_max = np.max(ratio)

                if ratio_max > ratio_thres or np.max(currDists_prevIds)/emd > A_tol: 
                    fNN_info[emd].append(True) # False nearest neighbor
                else:
                    fNN_info[emd].append(False)
                    
            # # calculate the percentage of points having false nearest neighbors
            # fNN_info[emd] = np.array(fNN_info[emd])
            # false_ratio = np.sum(fNN_info[emd])/len(fNN_info[emd])
            # if false_ratio < false_tol:
            #     return True, emd, false_ratio

    # if no embedding dimension is found, then return the one with the lowest false ratio
    false_ratio_list = []
    for emd in range(emd_min+emd_step, emd_max+1, emd_step):
        false_ratio = np.sum(fNN_info[emd])/len(fNN_info[emd])
        false_ratio_list.append(false_ratio)
    # get the index of the lowest false ratio
    min_false_idx = np.argmin(false_ratio_list)
    min_false_ratio = false_ratio_list[min_false_idx]
    min_false_emd = emd_min + (min_false_idx+1)*emd_step
    if min_false_ratio < false_tol:
        return True, min_false_emd, min_false_ratio
    else:
        return False, min_false_emd, min_false_ratio

