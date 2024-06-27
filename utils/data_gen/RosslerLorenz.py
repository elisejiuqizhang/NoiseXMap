# Unidirectionally coupled Rössler-Lorenz system (Section C of the paper "Nonuniform state-space reconstruction and coupling detection")
# Paper link: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.016207

# allowing options as follow:
# 1. noise type: None (deactivate all the following options if no noise), Gaussian, Laplacian
# 2. when to add noise: in-generation (process noise), post-generation (measurement noise)
# 3. how to add noise: additive (mean is 0), multiplicative (mean is 1), or both
# 4. noise level: a scalar that controls the noise intensity


# dX1/dt = 6*(-Y1-Z1)
# dY1/dt = 6*(X1+0.2*Y1)
# dZ1/dt = 6*(0.2+X1*Z1-5.7*Z1)

# dX2/dt = 10*(-Y2-Z2)
# dY2/dt = 28*X2-Y2-X2*Z2+C*Y1^2 (drive term governed by C)
# dZ2/dt = X2*Y2-8/3*Z2

# For the R-L system, another parameter is C, the coupling strength. The default value is 0.2.

import numpy as np
from scipy.integrate import odeint

# Define the Rössler-Lorenz system in raw form (no noise)
def RosslerLorenz_raw(X, t, C):
    x1, y1, z1, x2, y2, z2 = X
    dx1dt = 6*(-y1-z1)
    dy1dt = 6*(x1+0.2*y1)
    dz1dt = 6*(0.2+x1*z1-5.7*z1)
    dx2dt = 10*(-y2-z2)
    dy2dt = 28*x2-y2-x2*z2+C*y1**2
    dz2dt = x2*y2-8/3*z2
    return np.array([dx1dt, dy1dt, dz1dt, dx2dt, dy2dt, dz2dt])

# Define the Rössler-Lorenz system with in generation noise (process noise)
def RosslerLorenz_in(X, t, C, noiseType="Gaussian", noiseAddType="mult", noiseLevel=0.1):

    lp_add= np.random.laplace(0, noiseLevel, X.shape)
    g_add= np.random.normal(0, noiseLevel, X.shape)
    lp_mult= np.random.laplace(1, noiseLevel, X.shape)
    g_mult= np.random.normal(1, noiseLevel, X.shape)
        
    if noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
        if noiseAddType.lower()=='add' or 'additive':
            return RosslerLorenz_raw(X, C) + g_add
        elif noiseAddType.lower()=='mult' or 'multiplicative':
            return RosslerLorenz_raw(X, C) * g_mult
        elif noiseAddType.lower()=='both':
            return RosslerLorenz_raw(X, C) * g_mult + g_add
    elif noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
        if noiseAddType.lower()=='add' or 'additive':
            return RosslerLorenz_raw(X, C) + lp_add
        elif noiseAddType.lower()=='mult' or 'multiplicative':
            return RosslerLorenz_raw(X, C) * lp_mult
        elif noiseAddType.lower()=='both':
            return RosslerLorenz_raw(X, C) * lp_mult + lp_add
    else:
        raise ValueError("Noise type not recognized. Please choose 'Gaussian' or 'Laplacian'.")


# wrapper, generate data to a certain length L (default: 10000)
def gen_RosslerLorenz(C=0.2, noiseType=None, noiseWhen='in', noiseAddType="add", noiseLevel=0.1, L=10000):
    data = np.zeros((L+1, 6))
    # initial conditions
    data[0] = np.random.rand(6)
    dt=0.01
    t=np.linspace(0, L, int(L/dt))
    if noiseType == None or noiseType.lower()=='none':

        data = odeint(RosslerLorenz_raw, data[0], t, args=(C,))
        # # downsample to required length L
        # downstep=int(10000000/L)
        # data = data[::downstep]

    else: # with noise
        if noiseWhen.lower()=='post' or 'post-generation' or 'm' or 'measurement' or 'measurement-noise':
            
            data = odeint(RosslerLorenz_raw, data[0], t, args=(C,))
            # # downsample to required length L
            # downstep=int(10000000/L)
            # data = data[::downstep]

            lp_add= np.random.laplace(0, noiseLevel, data.shape)
            g_add= np.random.normal(0, noiseLevel, data.shape)
            lp_mult= np.random.laplace(1, noiseLevel, data.shape)
            g_mult= np.random.normal(1, noiseLevel, data.shape)
            if noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
                if noiseAddType.lower()=='add' or 'additive':
                    return data + g_add
                elif noiseAddType.lower()=='mult' or 'multiplicative':
                    return data * g_mult
                elif noiseAddType.lower()=='both':
                    return data * g_mult + g_add
            elif noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
                if noiseAddType.lower()=='add' or 'additive':
                    return data + lp_add
                elif noiseAddType.lower()=='mult' or 'multiplicative':
                    return data * lp_mult
                elif noiseAddType.lower()=='both':
                    return data * lp_mult + lp_add
            else:
                raise ValueError("Noise type not recognized. Please choose 'Gaussian' or 'Laplacian'.")
        elif noiseWhen.lower()=='in' or 'in-generation' or 'p' or 'process' or 'process-noise':
            
            data = odeint(RosslerLorenz_in, data[0], t, args=(C, noiseType, noiseAddType, noiseLevel))
            # # downsample to required length L
            # downstep=int(10000000/L)
            # data = data[::downstep]

        else:
            raise ValueError("Noise addition time not recognized. Please choose 'in' or 'post'.")

    return data


# # try to visualize the data to test
# import os, sys
# root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root)
# print(root)

# from XMap.DelayEmd import time_delay_embed
# import matplotlib.pyplot as plt
# data = gen_RosslerLorenz(C=0.2, noiseType='Gaussian', noiseWhen='in', noiseAddType='both', noiseLevel=0.1, L=10000)

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
#     plt.savefig('RosslerLorenz'+str(i)+'.png')