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


# Lorenz single variable noise injection - in-generation (process noise)
def Lorenz_in_singleV(xyz, *, s=10, r=28, b=2.667, noiseVar='x', noiseType='gNoise', noiseAddType=None, noiseLevel=None):
    """
    Parameters:
    -----------
    xyz : array-like, shape (3,)
        Point of interest in three-dimensional space.
    s, r, b : float
        Parameters defining the Lorenz attractor.
    noiseVar : str
        The variable to inject noise into, options: 'x', 'y', 'z' (ignore case while checking)
    noiseType : str
        Noises are either: "laplace"/'lpNoise'/'l'/'lap'/'lp' or "gaussian"/'gNoise'/'g' or "None"; (ignore case while checking)
    noiseAddType : str
        Noises are either: "mult", "add" or "both"; (ignore case while checking)
    noiseLevel : float
        The level of noise to be added; Only effective if noiseType is not "None";

    Returns:
    --------
    xyz_dot : array, shape (3,)
        Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z=xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    if noiseVar.lower()=='x':
        if noiseType.lower()=='laplace' or noiseType.lower()=='lpnoise' or noiseType.lower()=='lap' or noiseType.lower()=='lp' or noiseType.lower()=='l':
            if noiseAddType.lower()=='mult' or noiseAddType.lower()=='multiplicative':
                x_dot = x_dot*np.random.laplace(1, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                x_dot = x_dot+np.random.laplace(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='both':
                x_dot = x_dot*np.random.laplace(1, noiseLevel)+np.random.laplace(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
        elif noiseType.lower()=='gaussian' or noiseType.lower()=='gnoise' or noiseType.lower()=='gaus' or noiseType.lower()=='g':
            if noiseAddType.lower()=='mult' or noiseAddType.lower()=='multiplicative':
                x_dot = x_dot*np.random.normal(1, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                x_dot = x_dot+np.random.normal(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='both':
                x_dot = x_dot*np.random.normal(1, noiseLevel)+np.random.normal(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
    elif noiseVar.lower()=='y':
        if noiseType.lower()=='laplace' or noiseType.lower()=='lpnoise' or noiseType.lower()=='lap' or noiseType.lower()=='lp' or noiseType.lower()=='l':
            if noiseAddType.lower()=='mult' or noiseAddType.lower()=='multiplicative':
                y_dot = y_dot*np.random.laplace(1, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                y_dot = y_dot+np.random.laplace(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='both':
                y_dot = y_dot*np.random.laplace(1, noiseLevel)+np.random.laplace(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
        elif noiseType.lower()=='gaussian' or noiseType.lower()=='gnoise' or noiseType.lower()=='gaus' or noiseType.lower()=='g':
            if noiseAddType.lower()=='mult' or noiseAddType.lower()=='multiplicative':
                y_dot = y_dot*np.random.normal(1, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                y_dot = y_dot+np.random.normal(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='both':
                y_dot = y_dot*np.random.normal(1, noiseLevel)+np.random.normal(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
    elif noiseVar.lower()=='z':
        if noiseType.lower()=='laplace' or noiseType.lower()=='lpnoise' or noiseType.lower()=='lap' or noiseType.lower()=='lp' or noiseType.lower()=='l':
            if noiseAddType.lower()=='mult' or noiseAddType.lower()=='multiplicative':
                z_dot = z_dot*np.random.laplace(1, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                z_dot = z_dot+np.random.laplace(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='both':
                z_dot = z_dot*np.random.laplace(1, noiseLevel)+np.random.laplace(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
        elif noiseType.lower()=='gaussian' or noiseType.lower()=='gnoise' or noiseType.lower()=='gaus' or noiseType.lower()=='g':
            if noiseAddType.lower()=='mult' or noiseAddType.lower()=='multiplicative':
                z_dot = z_dot*np.random.normal(1, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                z_dot = z_dot+np.random.normal(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            elif noiseAddType.lower()=='both':
                z_dot = z_dot*np.random.normal(1, noiseLevel)+np.random.normal(0, noiseLevel)
                return np.array([x_dot, y_dot, z_dot])
            
# wrapper, generate data to a certain length L (default: 10000)
def gen_Lorenz_singleVarNoise(s=10, r=28, b=2.667, noiseVar='x', noiseType='gNoise', noiseWhen='in', noiseAddType="add", noiseLevel=0.1, L=10000):
    data = np.zeros((L+1, 3))
    
    dt = 0.01
    flag_restart = True # to start the iteration 
    
    if noiseWhen.lower()=="in" or noiseWhen.lower()=="in-generation":
        while flag_restart:
            flag_restart=False
            # initial conditions
            data[0] = np.random.rand(3)
            for i in range(L):
                data[i+1] = data[i] + Lorenz_in_singleV(data[i], s=s, r=r, b=b, noiseVar=noiseVar, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)*dt
                if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                    data[i+1] = np.random.rand(3)
                    flag_restart=True
                    break
        return data
    
    elif noiseWhen.lower()=='post' or noiseWhen.lower()=='post-generation':
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
        lp_add= np.random.laplace(0, noiseLevel, data.shape[0])
        g_add= np.random.normal(0, noiseLevel, data.shape[0])
        lp_mult= np.random.laplace(1, noiseLevel, data.shape[0])
        g_mult= np.random.normal(1, noiseLevel, data.shape[0])
        if noiseVar.lower()=='x':
            if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    data[:,0] = data[:,0]*lp_mult
                    return data
                elif noiseAddType.lower()=='add' or 'additive':
                    data[:,0] = data[:,0]+lp_add
                    return data
                elif noiseAddType.lower()=="both":
                    data[:,0] = data[:,0]*lp_mult+lp_add
                    return data
            elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    data[:,0] = data[:,0]*g_mult
                    return data
                elif noiseAddType.lower()=='add' or 'additive':
                    data[:,0] = data[:,0]+g_add
                    return data
                elif noiseAddType.lower()=="both":
                    data[:,0] = data[:,0]*g_mult+g_add
                    return data
        elif noiseVar.lower()=='y':
            if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    data[:,1] = data[:,1]*lp_mult
                    return data
                elif noiseAddType.lower()=='add' or 'additive':
                    data[:,1] = data[:,1]+lp_add
                    return data
                elif noiseAddType.lower()=="both":
                    data[:,1] = data[:,1]*lp_mult+lp_add
                    return data
            elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    data[:,1] = data[:,1]*g_mult
                    return data
                elif noiseAddType.lower()=='add' or 'additive':
                    data[:,1] = data[:,1]+g_add
                    return data
                elif noiseAddType.lower()=="both":
                    data[:,1] = data[:,1]*g_mult+g_add
                    return data
        elif noiseVar.lower()=='z':
            if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    data[:,2] = data[:,2]*lp_mult
                    return data
                elif noiseAddType.lower()=='add' or 'additive':
                    data[:,2] = data[:,2]+lp_add
                    return data
                elif noiseAddType.lower()=="both":
                    data[:,2] = data[:,2]*lp_mult+lp_add
                    return data
            elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
                if noiseAddType.lower()=="mult" or 'multiplicative':
                    data[:,2] = data[:,2]*g_mult
                    return data
                elif noiseAddType.lower()=='add' or 'additive':
                    data[:,2] = data[:,2]+g_add
                    return data
                elif noiseAddType.lower()=="both":
                    data[:,2] = data[:,2]*g_mult+g_add
                    return data


# # try to visualize the data to test
# import os, sys
# root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root)
# print(root)

# from XMap.DelayEmd import time_delay_embed
# import matplotlib.pyplot as plt

# L=10000

# # data = gen_Lorenz(noiseType='l', noiseWhen='in', noiseAddType='both', noiseLevel=0.6, L=L)
# data = gen_Lorenz_singleVarNoise(noiseVar='z', noiseType='g', noiseWhen='in', noiseAddType='both', noiseLevel=0.8, L=L)

# plot_L = 2000
# tau=2
# emd=3

# # loop over all variables
# for i in range(data.shape[1]):
#     data_emd=time_delay_embed(data[int(L/3*2+1):int(L/3*2+1)+plot_L,i], tau, emd)
#     # 3d plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(data_emd[:,0], data_emd[:,1], data_emd[:,2])
#     plt.show()
#     plt.savefig('Lorenz'+str(i)+'.png')