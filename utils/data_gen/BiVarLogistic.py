# Bivariate logistic map - can be considered as two species interactions
# X(t+1)=X(t)*(a_x*(1-X(t)) - b_yx*Y(t))
# Y(t+1)=Y(t)*(a_y*(1-Y(t)) - b_xy*X(t))

import numpy as np
import random

# in-generation noise
def BiVarLogistic_in(X, a_x=3.8, a_y=3.5, b_xy=0.3, b_yx=0.3, noiseType=None, noiseAddType="add", noiseLevel=0.1):
    x, y = X
    x_next = x*(a_x*(1-x) - b_yx*y)
    y_next = y*(a_y*(1-y) - b_xy*x)

    if noiseType==None or noiseType.lower()=="none":
        return np.array([x_next, y_next])
    
    elif noiseType.lower()=='l' or noiseType.lower()=='laplacian' or noiseType.lower()=='lap' or noiseType.lower()=='lpnoise':
        lp_add= np.random.laplace(0, noiseLevel, X.shape)
        lp_mult= np.random.laplace(1, noiseLevel, X.shape)
        if noiseAddType.lower()=="mult" or noiseAddType.lower()=='multiplicative':
            return np.array([x_next, y_next])*lp_mult
        elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
            return np.array([x_next, y_next])+lp_add
        elif noiseAddType.lower()=="both":
            return np.array([x_next, y_next]*lp_mult+lp_add)
        
    elif noiseType.lower()=='g' or noiseType.lower()=='gaussian' or noiseType.lower()=='gaus' or noiseType.lower()=='gnoise':
        g_mult= np.random.normal(1, noiseLevel, X.shape)
        g_add= np.random.normal(0, noiseLevel, X.shape)
        if noiseAddType.lower()=="mult" or noiseAddType.lower()=='multiplicative':
            return np.array([x_next, y_next])*g_mult
        elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
            return np.array([x_next, y_next])+g_add
        elif noiseAddType.lower()=="both":
            return np.array([x_next, y_next]*g_mult+g_add)

            
# wrapper, generate data to a certain length L (default: 10000)
def gen_BiVarLogistic(a_x=3.7, a_y=3.72, b_xy=0.35, b_yx=0.32, noiseType=None, noiseWhen='in', noiseAddType="add", noiseLevel=0.1, L=10000):
    data = np.zeros((L+1, 2))
    # initial conditions
    # data[0] = np.array([0.2, 0.4])
    # data[0]=np.random.rand(2)

    if noiseType==None or noiseType.lower()=="none":
        # for i in range(L):
            # data[i+1] = BiVarLogistic_in(data[i], a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx)
        flag_rerun=True
        while(flag_rerun):
            # sample initial conditions
            data[0]=np.random.rand(2)
            flag_rerun=False
            for i in range(L):
                data[i+1] = BiVarLogistic_in(data[i], a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=None)
                # check if there is divergence or nan values
                if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                    flag_rerun=True
                    break
        return data

    else: # with noise
        if noiseWhen.lower()=="in" or noiseWhen.lower()=="in-generation":
            # for i in range(L):
                # data[i+1] = BiVarLogistic_in(data[i], a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)
            flag_rerun=True
            while(flag_rerun):
                # sample initial conditions
                flag_rerun=False
                data[0]=np.random.rand(2)
                for i in range(L):
                    data[i+1] = BiVarLogistic_in(data[i], a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)
                    # check if there is divergence or nan values
                    if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                        flag_rerun=True
                        break
            return data

        elif noiseWhen.lower()=="post" or noiseWhen.lower()=="post-generation":
            # for i in range(L):
                # data[i+1] = BiVarLogistic_in(data[i], a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=None)
            
            flag_rerun=True
            while(flag_rerun):
                # sample initial conditions
                data[0]=np.random.rand(2)
                flag_rerun=False
                for i in range(L):
                    data[i+1] = BiVarLogistic_in(data[i], a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=None)
                    # check if there is divergence or nan values
                    if np.isnan(data[i+1]).any() or np.isinf(data[i+1]).any():
                        flag_rerun=True
                        break
            
            lp_add= np.random.laplace(0, noiseLevel, data.shape)
            g_add= np.random.normal(0, noiseLevel, data.shape)
            lp_mult= np.random.laplace(1, noiseLevel, data.shape)
            g_mult= np.random.normal(1, noiseLevel, data.shape)
            
            if noiseType.lower()=='l' or noiseType.lower()=='laplacian' or noiseType.lower()=='lap' or noiseType.lower()=='lpNoise':
                if noiseAddType.lower()=="mult" or noiseAddType.lower()=='multiplicative':
                    data=data*lp_mult
                elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                    data=data+lp_add
                elif noiseAddType.lower()=="both":
                    data=data*lp_mult+lp_add
            elif noiseType.lower()=='g' or noiseType.lower()=='gaussian' or noiseType.lower()=='gaus' or noiseType.lower()=='gNoise':
                if noiseAddType.lower()=="mult" or noiseAddType.lower()=='multiplicative':
                    data=data*g_mult
                elif noiseAddType.lower()=='add' or noiseAddType.lower()=='additive':
                    data=data+g_add
                elif noiseAddType.lower()=="both":
                    data=data*g_mult+g_add
    return data


# # try to visualize the data to test
# import os, sys
# root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root)
# print(root)

# from XMap.DelayEmd import time_delay_embed
# import matplotlib.pyplot as plt
# data = gen_BiVarLogistic(noiseType='l', noiseWhen='in', noiseAddType='both', noiseLevel=0.001, L=10000)

# # 2D
# plot_L = 10000
# tau=2
# emd=2
# # loop over all variables
# for i in range(data.shape[1]):
#     data_emd=time_delay_embed(data[:plot_L,i], tau, emd)
#     # 3d plot
#     plt.plot(data_emd[:,0], data_emd[:,1])
#     plt.show()
#     plt.savefig('BiVar'+str(i)+'.png')
#     plt.close()

# # 3D
# plot_L = 10000
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
#     plt.savefig('BiVar'+str(i)+'.png')
#     plt.close()