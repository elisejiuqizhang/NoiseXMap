# visualize the bifurcation diagram of the logistic map

import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)


import numpy as np
import random
import matplotlib.pyplot as plt
import imageio # generate gif that can loop out of the pngs

from utils.data_gen.BiVarLogistic import BiVarLogistic_in

# set random seed for reproducibility
random.seed(97)
np.random.seed(97)

output_dir = os.path.join(root, 'outputs', 'viz', 'attractors', 'BiVarLogistic')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

L = 30000000


def logistic_vizBif_ax2x(X, a_x, a_y, b_xy=0, b_yx=0, noiseType=None, noiseWhen='in', noiseAddType="add", noiseLevel=0.01, a_x_step=1e-7):
    if noiseWhen.lower()=="in" or "in-generation":
        X_next = BiVarLogistic_in(X, a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)
    elif noiseWhen.lower()=="post" or "post-generation":
        X_next = BiVarLogistic_in(X, a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=None)
        lp_add= np.random.laplace(0, noiseLevel, X.shape)
        g_add= np.random.normal(0, noiseLevel, X.shape)
        lp_mult= np.random.laplace(1, noiseLevel, X.shape)
        g_mult= np.random.normal(1, noiseLevel, X.shape)
        if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
            if noiseAddType.lower()=="mult" or 'multiplicative':
                X_next = X_next*lp_mult
            elif noiseAddType.lower()=='add' or 'additive':
                X_next = X_next+lp_add
            elif noiseAddType.lower()=="both":
                X_next = X_next*lp_mult+lp_add
        elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
            if noiseAddType.lower()=="mult" or 'multiplicative':
                X_next = X_next*g_mult
            elif noiseAddType.lower()=='add' or 'additive':
                X_next = X_next+g_add
            elif noiseAddType.lower()=="both":
                X_next = X_next*g_mult+g_add
    a_x_next = a_x + a_x_step
    yield a_x_next, X_next

def logistic_vizBif_ay2y(X, a_y, a_x, b_xy=0, b_yx=0, noiseType=None, noiseWhen='in', noiseAddType="add", noiseLevel=0.01, a_y_step=1e-7):
    if noiseWhen.lower()=="in" or "in-generation":
        X_next = BiVarLogistic_in(X, a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)
    elif noiseWhen.lower()=="post" or "post-generation":
        X_next = BiVarLogistic_in(X, a_x=a_x, a_y=a_y, b_xy=b_xy, b_yx=b_yx, noiseType=None)
        lp_add= np.random.laplace(0, noiseLevel, X.shape)
        g_add= np.random.normal(0, noiseLevel, X.shape)
        lp_mult= np.random.laplace(1, noiseLevel, X.shape)
        g_mult= np.random.normal(1, noiseLevel, X.shape)
        if noiseType.lower()=='l' or 'laplacian' or 'lap' or 'l' or 'lpNoise':
            if noiseAddType.lower()=="mult" or 'multiplicative':
                X_next = X_next*lp_mult
            elif noiseAddType.lower()=='add' or 'additive':
                X_next = X_next+lp_add
            elif noiseAddType.lower()=="both":
                X_next = X_next*lp_mult+lp_add
        elif noiseType.lower()=='g' or 'gaussian' or 'gaus' or 'gNoise':
            if noiseAddType.lower()=="mult" or 'multiplicative':
                X_next = X_next*g_mult
            elif noiseAddType.lower()=='add' or 'additive':
                X_next = X_next+g_add
            elif noiseAddType.lower()=="both":
                X_next = X_next*g_mult+g_add
    a_y_next = a_y + a_y_step
    yield a_y_next, X_next

# visualize bifurcation diagram of the BiVarLogistic attractor
list_noiseType=[None, 'Gaussian', 'Laplacian']
# list_noiseType=['Gaussian', 'Laplacian']
# list_noiseType=[None]
list_noiseWhen=['in', 'post']
list_noiseAddType=['add', 'mult', 'both']   
list_noiseLevel=[1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6]

for noiseType in list_noiseType:
    if noiseType==None or noiseType.lower()=='none':
        rate_start=2.6
        rate_end=4
        step = (rate_end - rate_start) / L

        noiseType_str = 'None'
        imgs_dir = os.path.join(output_dir, noiseType_str)
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        # generate non-noisy data
        A_x=np.zeros(L+1)
        X=np.zeros((L+1, 2))
        A_x[0]=rate_start
        X[0]=np.array([0.5, 0.5])
        for i in range(L):
            ax_next, X_next = next(logistic_vizBif_ax2x(X[i], a_x=A_x[i], a_y=6.5-A_x[i], b_xy=0, b_yx=0, noiseType=noiseType, a_x_step=step))
            A_x[i+1] = ax_next
            X[i+1] = X_next

        # bifurcation diagram
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 10))
        plt.plot(A_x, X[:,0], '^', color='white', alpha=0.4, markersize = 0.013)
        plt.axis('on')
        plt.x_label = 'a_x'
        plt.y_label = 'x'
        plt.title('noiseType: '+noiseType_str)
        plt.savefig(os.path.join(imgs_dir,noiseType_str+'.png'))
        plt.close()

    else: # noise cases
        rate_start=2.6
        rate_end=4
        step = (rate_end - rate_start) / L

        gifs_dir = os.path.join(output_dir, 'gifs')
        if not os.path.exists(gifs_dir):
            os.makedirs(gifs_dir)

        noiseType_str = noiseType
        imgs_dir = os.path.join(output_dir, noiseType_str)
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        for noiseWhen in list_noiseWhen:
            for noiseAddType in list_noiseAddType:
                for noiseLevel in list_noiseLevel:
                    case_save_dir = os.path.join(imgs_dir,noiseWhen,noiseAddType)
                    if not os.path.exists(case_save_dir):
                        os.makedirs(case_save_dir)
                    A_x=np.zeros(L+1)
                    X=np.zeros((L+1, 2))
                    A_x[0]=rate_start
                    X[0]=np.array([0.5, 0.5])
                    for i in range(L):
                        ax_next, X_next = next(logistic_vizBif_ax2x(X[i], a_x=A_x[i], a_y=6.5-A_x[i], b_xy=0, b_yx=0, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel, a_x_step=step))
                        A_x[i+1] = ax_next
                        X[i+1] = X_next

                    # bifurcation diagram
                    plt.style.use('dark_background')
                    plt.figure(figsize=(10, 10))
                    plt.plot(A_x, X[:,0], '^', color='white', alpha=0.4, markersize = 0.013)
                    plt.axis('on')
                    plt.x_label = 'a_x'
                    plt.y_label = 'x'
                    plt.title('noiseType: '+noiseType_str+'\nnoiseWhen: '+noiseWhen+'\nnoiseAddType: '+noiseAddType+'\nnoiseLevel: '+str(noiseLevel))
                    plt.savefig(os.path.join(case_save_dir,str(noiseLevel)+'.png'))
                    plt.close()

                # generate gif that loops forever
                images = [imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'.png')) for noiseLevel in list_noiseLevel]
                imageio.mimsave(os.path.join(gifs_dir,noiseType_str+'_'+noiseWhen+'_'+noiseAddType+'.gif'), images, duration=400, loop=0)