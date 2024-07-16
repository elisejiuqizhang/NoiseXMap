import numpy as np
import random

# in-generation noise
def Lorenz_in(xyz, *, s=10, r=28, b=2.667, noiseType="None", noiseAddType=None, noiseLevel=None):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.
    noiseType: str
        Noises are either: "laplace" or "gaussian" or "None"; (ignore case while checking)
    noiseAddType: str
        Noises are either: "mult", "add" or "both"; Only effective if noiseType is not "None"; (ignore case while checking)
    noiseLevel: float
        The level of noise to be added; Only effective if noiseType is not "None";

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    if noiseType==None or noiseType.lower()=="none":
        return np.array([x_dot, y_dot, z_dot])
    else:
        lp_add= np.random.laplace(0, noiseLevel, xyz.shape)
        g_add= np.random.normal(0, noiseLevel, xyz.shape)
        lp_mult= np.random.laplace(1, noiseLevel, xyz.shape)
        g_mult= np.random.normal(1, noiseLevel, xyz.shape)
        if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
            if noiseAddType.lower()=="mult" or 'multiplicative':
                return np.array([x_dot, y_dot, z_dot])*lp_mult
            elif noiseAddType.lower()=='add' or 'additive':
                return np.array([x_dot, y_dot, z_dot])+lp_add
            elif noiseAddType.lower()=="both":
                return np.array([x_dot, y_dot, z_dot]*lp_mult+lp_add)
        elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
            if noiseAddType.lower()=="mult" or 'multiplicative':
                return np.array([x_dot, y_dot, z_dot])*g_mult
            elif noiseAddType.lower()=='add' or 'additive':
                return np.array([x_dot, y_dot, z_dot])+g_add
            elif noiseAddType.lower()=="both":
                return np.array([x_dot, y_dot, z_dot]*g_mult+g_add)

# wrapper, generate data to a certain length L (default: 10000)
def gen_Lorenz(s=10, r=28, b=2.667, noiseType=None, noiseWhen='in', noiseAddType="add", noiseLevel=0.1, L=10000):
    data = np.zeros((L+1, 3))

    dt = 0.01
    flag_restart = True # to start the iteration 

    if noiseType==None or noiseType.lower()=="none":
        # for i in range(L):
        #     data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b)*dt
        while flag_restart:
            flag_restart=False
            # initial conditions
            data[0] = np.random.rand(3)
            for i in range(L):
                data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)*dt
                if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                    data[i+1] = np.random.rand(3)
                    flag_restart=True
                    break
                            
    else: # with noise
        if noiseWhen.lower()=="in" or "in-generation":
            # for i in range(L):
            #     data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)*dt
            while flag_restart:
                flag_restart=False
                # initial conditions
                data[0] = np.random.rand(3)
                for i in range(L):
                    data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)*dt
                    if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                        data[i+1] = np.random.rand(3)
                        flag_restart=True
                        break
 

        elif noiseWhen.lower()=="post" or "post-generation":
            # for i in range(L):
            #     data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b)*dt
            while flag_restart:
                flag_restart=False
                # initial conditions
                data[0] = np.random.rand(3)
                for i in range(L):
                    data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b)*dt
                    if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                        data[i+1] = np.random.rand(3)
                        flag_restart=True
                        break
            lp_add= np.random.laplace(0, noiseLevel, data.shape)
            g_add= np.random.normal(0, noiseLevel, data.shape)
            lp_mult= np.random.laplace(1, noiseLevel, data.shape)
            g_mult= np.random.normal(1, noiseLevel, data.shape)
            if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    return data*lp_mult
                elif noiseAddType.lower()=='add' or 'additive':
                    return data+lp_add
                elif noiseAddType.lower()=="both":
                    return data*lp_mult+lp_add
            elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    return data*g_mult
                elif noiseAddType.lower()=='add' or 'additive':
                    return data+g_add
                elif noiseAddType.lower()=="both":
                    return data*g_mult+g_add
    return data

# # try to visualize the data to test
# import os, sys
# root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root)
# print(root)

# from XMap.DelayEmd import time_delay_embed
# import matplotlib.pyplot as plt
# data = gen_Lorenz(noiseType='l', noiseWhen='in', noiseAddType='both', noiseLevel=0.6, L=10000)

# plot_L = 1000
# tau=2
# emd=3

# # loop over all variables
# for i in range(data.shape[1]):
#     data_emd=time_delay_embed(data[:plot_L,i], tau, emd)
#     # 3d plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(data_emd[:,0], data_emd[:,1], data_emd[:,2])
#     plt.show()
#     plt.savefig('Lorenz'+str(i)+'.png')