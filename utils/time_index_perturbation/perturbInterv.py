import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

import numpy as np
import random
import pandas as pd

# randomly swap order of datapoints within each interval

def perturbInterv(data, lenInterv=5, percentInterv=0.5, swapPercent=0.5):
    """
    Randomly swap order of points within selected intervals, return the swapped time series.
    Input:
        data (pd.DataFrame or np.ndarray): time series data, can be multivar;
        lenInterv (int): length of interval to swap;
        percentInterv (float): percentage of intervals to swap;
        
    Output:
        data_perturb (pd.DataFrame or np.ndarray): perturbed time series data;
    """

    data_perturb=data.copy()

    # length
    L=data.shape[0]

    # check if input is legal (lenInterv should be smaller than half of totalL; percent between 0 and 1 exclusive)
    if lenInterv>=L/2 or lenInterv<=2:
        raise ValueError('lenInterv should be smaller than half of totalL and greater than 2')
    if percentInterv<=0 or percentInterv>=1:
        raise ValueError('percentInterv should be between 0 and 1 exclusive')
    if swapPercent<=0 or swapPercent>=1:
        raise ValueError('swapPercent should be between 0 and 1 exclusive')
    
    # total number of intervals - keep the last one (if not full length) intact
    numInterv=int(L/lenInterv)
    # number of intervals to swap
    numSwapInterv=int(numInterv*percentInterv)
    # number of points to swap within each interval
    numSwapPoints=int(lenInterv*swapPercent)
    if numSwapPoints<=1:
        numSwapPoints=2 # to make sure there is at least a swap going on

    # randomly select intervals to swap, and swap the num of points as defined
    # generate #numSwapInterv random numbers in [0, numInterv-1]
    swapIntervs=random.sample(range(numInterv), numSwapInterv)
    for intervStart in range(numInterv):
        if intervStart in swapIntervs: # this is the interval to operate on
            # randomly sample #numSwapPoints indices - they are going to be perturnbed
            swapPoints=random.sample(range(intervStart*lenInterv, (intervStart+1)*lenInterv), numSwapPoints)
            # randomly mess up the orders of these points
            shuffled_swapIndc=random.sample(swapPoints, len(swapPoints))
            # assign the shuffled points back to the original data
            if isinstance(data, pd.DataFrame):
                data_perturb.iloc[swapPoints]=data.iloc[shuffled_swapIndc]
            elif isinstance(data, np.ndarray):
                data_perturb[swapPoints]=data[shuffled_swapIndc]

    return data_perturb




# list_nums=np.arange(1,51)

# # bivar df
# df=pd.DataFrame({'X':list_nums, 'Y':list_nums})

# # test 1: with pd.DataFrame [1,...,50]
# df_perturb=perturbInterv(df, lenInterv=5, percentInterv=0.6, swapPercent=0.6)

# # test 2: with np.ndarray [1,...,50]
# arr=np.array(list_nums)
# arr_perturb=perturbInterv(arr, lenInterv=5, percentInterv=0.6, swapPercent=0.6)

# print(df_perturb)
# print(arr_perturb)