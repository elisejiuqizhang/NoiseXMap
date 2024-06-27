import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from utils.data_gen.Lorenz import *
from utils.XMap.DelayEmd import *

# k nearest neighbors help
from sklearn.neighbors import NearestNeighbors

import numpy as np
import random
import matplotlib.pyplot as plt
import imageio # generate gif that can loop out of the pngs

# set random seed for reproducibility
random.seed(97)
np.random.seed(97)

output_dir = os.path.join(root, 'outputs', 'viz', 'XMapDE', 'Lorenz')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


n_neigh=14
tau=2
emd=3

L = 10000
plot_L = 10000
start_plot=np.random.randint(0, L-plot_L) if L>plot_L else 0
centroid_idx=np.random.randint(0, plot_L-1)

# range of noise levels to test
list_noiseType=[None, 'Gaussian', 'Laplacian']
# list_noiseType=[None]
# list_noiseType=['Gaussian', 'Laplacian']
list_noiseWhen=['in', 'post']
list_noiseAddType=['add', 'mult', 'both']   
list_noiseLevel=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

for noiseType in list_noiseType:
    if noiseType==None or noiseType.lower()=='none': # no need for gif in this case
        noiseType_str = 'None'
        imgs_dir = os.path.join(output_dir, noiseType_str)
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        # generate non-noisy data
        data=gen_Lorenz(noiseType=noiseType, L=L)
        data=data[start_plot:start_plot+plot_L,:]

        # get each variable's delay embedding
        data_X=data[:,0]
        data_Y=data[:,1]
        data_Z=data[:,2]
        X_embed=time_delay_embed(data_X, tau, emd)
        Y_embed=time_delay_embed(data_Y, tau, emd)
        Z_embed=time_delay_embed(data_Z, tau, emd)

        # find k nearest neighbors on X_embed
        neigh = NearestNeighbors(n_neighbors=n_neigh)
        neigh.fit(X_embed)
        _, indicesX = neigh.kneighbors(X_embed[centroid_idx].reshape(1, -1))

        # find k nearest neighbors on Y_embed
        neigh = NearestNeighbors(n_neighbors=n_neigh)
        neigh.fit(Y_embed)
        _, indicesY = neigh.kneighbors(Y_embed[centroid_idx].reshape(1, -1))

        # find k nearest neighbors on Z_embed
        neigh = NearestNeighbors(n_neighbors=n_neigh)
        neigh.fit(Z_embed)
        _, indicesZ = neigh.kneighbors(Z_embed[centroid_idx].reshape(1, -1))

        # 3D plot - order from left to right: X, Y, Z
        # three different pngs for each case, by the nn indices used (indicesX, indicesY, indicesZ)
        # using indicesX to plot
        fig=plt.figure(figsize=(36,12))
        ax1=fig.add_subplot(131, projection='3d')
        ax1.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax1.scatter(X_embed[indicesX[0][i],0], X_embed[indicesX[0][i],1], X_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
        ax1.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
        ax1.set_title('X')
        ax1.set_xlabel('X(t)')
        ax1.set_ylabel('X(t-tau)')
        ax1.set_zlabel('X(t-2*tau)')
        ax1.set_xlim(-20,20)
        ax1.set_ylim(-35,35)
        ax1.set_zlim(-5,25)

        ax2=fig.add_subplot(132, projection='3d')
        ax2.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax2.scatter(Y_embed[indicesX[0][i],0], Y_embed[indicesX[0][i],1], Y_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
        ax2.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
        ax2.set_title('Y')
        ax2.set_xlabel('Y(t)')
        ax2.set_ylabel(f'Y(t-{tau})')
        ax2.set_zlabel(f'Y(t-{2*tau})')
        ax2.set_xlim(-30,30)
        ax2.set_ylim(-40,40)
        ax2.set_zlim(-5,40)

        ax3=fig.add_subplot(133, projection='3d')
        ax3.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax3.scatter(Z_embed[indicesX[0][i],0], Z_embed[indicesX[0][i],1], Z_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
        ax3.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
        ax3.set_title('Z')
        ax3.set_xlabel('Z(t)')
        ax3.set_ylabel(f'Z(t-{tau})')
        ax3.set_zlabel(f'Z(t-{2*tau})')
        ax3.set_xlim(-3,60)
        ax3.set_ylim(-3,60)
        ax3.set_zlim(-3,60)

        fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing X to map back to Y and Z')
        plt.savefig(os.path.join(imgs_dir,noiseType_str+'X2YZ.png'))
        plt.close()

        # using indicesY to plot
        fig=plt.figure(figsize=(36,12))
        ax1=fig.add_subplot(131, projection='3d')
        ax1.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax1.scatter(X_embed[indicesY[0][i],0], X_embed[indicesY[0][i],1], X_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
        ax1.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
        ax1.set_title('X')
        ax1.set_xlabel('X(t)')
        ax1.set_ylabel('X(t-tau)')
        ax1.set_zlabel('X(t-2*tau)')
        ax1.set_xlim(-20,20)
        ax1.set_ylim(-35,35)
        ax1.set_zlim(-5,25)

        ax2=fig.add_subplot(132, projection='3d')
        ax2.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax2.scatter(Y_embed[indicesY[0][i],0], Y_embed[indicesY[0][i],1], Y_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
        ax2.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
        ax2.set_title('Y')
        ax2.set_xlabel('Y(t)')
        ax2.set_ylabel(f'Y(t-{tau})')
        ax2.set_zlabel(f'Y(t-{2*tau})')
        ax2.set_xlim(-30,30)
        ax2.set_ylim(-40,40)
        ax2.set_zlim(-5,40)

        ax3=fig.add_subplot(133, projection='3d')
        ax3.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax3.scatter(Z_embed[indicesY[0][i],0], Z_embed[indicesY[0][i],1], Z_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
        ax3.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
        ax3.set_title('Z')
        ax3.set_xlabel('Z(t)')
        ax3.set_ylabel(f'Z(t-{tau})')
        ax3.set_zlabel(f'Z(t-{2*tau})')
        ax3.set_xlim(-3,60)
        ax3.set_ylim(-3,60)
        ax3.set_zlim(-3,60)

        fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Y to map back to X and Z')
        plt.savefig(os.path.join(imgs_dir,noiseType_str+'Y2XZ.png'))
        plt.close()

        # using indicesZ to plot
        fig=plt.figure(figsize=(36,12))
        ax1=fig.add_subplot(131, projection='3d')
        ax1.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax1.scatter(X_embed[indicesZ[0][i],0], X_embed[indicesZ[0][i],1], X_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
        ax1.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
        ax1.set_title('X')
        ax1.set_xlabel('X(t)')
        ax1.set_ylabel('X(t-tau)')
        ax1.set_zlabel('X(t-2*tau)')
        ax1.set_xlim(-20,20)
        ax1.set_ylim(-35,35)
        ax1.set_zlim(-5,25)

        ax2=fig.add_subplot(132, projection='3d')
        ax2.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax2.scatter(Y_embed[indicesZ[0][i],0], Y_embed[indicesZ[0][i],1], Y_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
        ax2.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
        ax2.set_title('Y')
        ax2.set_xlabel('Y(t)')
        ax2.set_ylabel(f'Y(t-{tau})')
        ax2.set_zlabel(f'Y(t-{2*tau})')
        ax2.set_xlim(-30,30)
        ax2.set_ylim(-40,40)
        ax2.set_zlim(-5,40)

        ax3=fig.add_subplot(133, projection='3d')
        ax3.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
        for i in range(n_neigh):
            ax3.scatter(Z_embed[indicesZ[0][i],0], Z_embed[indicesZ[0][i],1], Z_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
        ax3.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
        ax3.set_title('Z')
        ax3.set_xlabel('Z(t)')
        ax3.set_ylabel(f'Z(t-{tau})')
        ax3.set_zlabel(f'Z(t-{2*tau})')
        ax3.set_xlim(-3,60)
        ax3.set_ylim(-3,60)
        ax3.set_zlim(-3,60)

        fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Z to map back to X and Y')
        plt.savefig(os.path.join(imgs_dir,noiseType_str+'Z2XY.png'))
        plt.close()


    else: # noise cases
        # create gifs for each noise case
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

                    data=gen_Lorenz(noiseType=noiseType, noiseWhen=noiseWhen, noiseAddType=noiseAddType, noiseLevel=noiseLevel, L=L)
                    data=data[start_plot:start_plot+plot_L,:]

                    # get each variable's delay embedding
                    data_X=data[:,0]
                    data_Y=data[:,1]
                    data_Z=data[:,2]
                    X_embed=time_delay_embed(data_X, tau, emd)
                    Y_embed=time_delay_embed(data_Y, tau, emd)
                    Z_embed=time_delay_embed(data_Z, tau, emd)

                    # find k nearest neighbors on X_embed
                    neigh = NearestNeighbors(n_neighbors=n_neigh)
                    neigh.fit(X_embed)
                    _, indicesX = neigh.kneighbors(X_embed[centroid_idx].reshape(1, -1))

                    # find k nearest neighbors on Y_embed
                    neigh = NearestNeighbors(n_neighbors=n_neigh)
                    neigh.fit(Y_embed)
                    _, indicesY = neigh.kneighbors(Y_embed[centroid_idx].reshape(1, -1))

                    # find k nearest neighbors on Z_embed
                    neigh = NearestNeighbors(n_neighbors=n_neigh)
                    neigh.fit(Z_embed)
                    _, indicesZ = neigh.kneighbors(Z_embed[centroid_idx].reshape(1, -1))

                    # 3D plot - order from left to right: X, Y, Z
                    # three different pngs for each case, by the nn indices used (indicesX, indicesY, indicesZ)
                    # using indicesX to plot
                    fig=plt.figure(figsize=(36,12))
                    ax1=fig.add_subplot(131, projection='3d')
                    ax1.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax1.scatter(X_embed[indicesX[0][i],0], X_embed[indicesX[0][i],1], X_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
                    ax1.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
                    ax1.set_title('X')
                    ax1.set_xlabel('X(t)')
                    ax1.set_ylabel('X(t-tau)')
                    ax1.set_zlabel('X(t-2*tau)')
                    ax1.set_xlim(-20,20)
                    ax1.set_ylim(-35,35)
                    ax1.set_zlim(-5,25)

                    ax2=fig.add_subplot(132, projection='3d')
                    ax2.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax2.scatter(Y_embed[indicesX[0][i],0], Y_embed[indicesX[0][i],1], Y_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
                    ax2.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
                    ax2.set_title('Y')
                    ax2.set_xlabel('Y(t)')
                    ax2.set_ylabel(f'Y(t-{tau})')
                    ax2.set_zlabel(f'Y(t-{2*tau})')
                    ax2.set_xlim(-30,30)
                    ax2.set_ylim(-40,40)
                    ax2.set_zlim(-5,40)

                    ax3=fig.add_subplot(133, projection='3d')
                    ax3.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax3.scatter(Z_embed[indicesX[0][i],0], Z_embed[indicesX[0][i],1], Z_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
                    ax3.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
                    ax3.set_title('Z')
                    ax3.set_xlabel('Z(t)')
                    ax3.set_ylabel(f'Z(t-{tau})')
                    ax3.set_zlabel(f'Z(t-{2*tau})')
                    ax3.set_xlim(-3,60)
                    ax3.set_ylim(-3,60)
                    ax3.set_zlim(-3,60)

                    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing X to map back to Y and Z'+'\nnoiseWhen: '+noiseWhen+'\nnoiseAddType: '+noiseAddType+'\nnoiseLevel: '+str(noiseLevel))
                    plt.savefig(os.path.join(case_save_dir,str(noiseLevel)+'X2YZ.png'))
                    plt.close()

                    # using indicesY to plot
                    fig=plt.figure(figsize=(36,12))
                    ax1=fig.add_subplot(131, projection='3d')
                    ax1.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax1.scatter(X_embed[indicesY[0][i],0], X_embed[indicesY[0][i],1], X_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
                    ax1.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
                    ax1.set_title('X')
                    ax1.set_xlabel('X(t)')
                    ax1.set_ylabel('X(t-tau)')
                    ax1.set_zlabel('X(t-2*tau)')
                    ax1.set_xlim(-20,20)
                    ax1.set_ylim(-35,35)
                    ax1.set_zlim(-5,25)

                    ax2=fig.add_subplot(132, projection='3d')
                    ax2.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax2.scatter(Y_embed[indicesY[0][i],0], Y_embed[indicesY[0][i],1], Y_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
                    ax2.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
                    ax2.set_title('Y')
                    ax2.set_xlabel('Y(t)')
                    ax2.set_ylabel(f'Y(t-{tau})')
                    ax2.set_zlabel(f'Y(t-{2*tau})')
                    ax2.set_xlim(-30,30)
                    ax2.set_ylim(-40,40)
                    ax2.set_zlim(-5,40)

                    ax3=fig.add_subplot(133, projection='3d')
                    ax3.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax3.scatter(Z_embed[indicesY[0][i],0], Z_embed[indicesY[0][i],1], Z_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
                    ax3.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
                    ax3.set_title('Z')
                    ax3.set_xlabel('Z(t)')
                    ax3.set_ylabel(f'Z(t-{tau})')
                    ax3.set_zlabel(f'Z(t-{2*tau})')
                    ax3.set_xlim(-3,60)
                    ax3.set_ylim(-3,60)
                    ax3.set_zlim(-3,60)

                    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Y to map back to X and Z'+ '\nnoiseWhen: '+noiseWhen+'\nnoiseAddType: '+noiseAddType+'\nnoiseLevel: '+str(noiseLevel))
                    plt.savefig(os.path.join(case_save_dir,str(noiseLevel)+'Y2XZ.png'))
                    plt.close()

                    # using indicesZ to plot
                    fig=plt.figure(figsize=(36,12))
                    ax1=fig.add_subplot(131, projection='3d')
                    ax1.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax1.scatter(X_embed[indicesZ[0][i],0], X_embed[indicesZ[0][i],1], X_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
                    ax1.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
                    ax1.set_title('X')
                    ax1.set_xlabel('X(t)')
                    ax1.set_ylabel('X(t-tau)')
                    ax1.set_zlabel('X(t-2*tau)')
                    ax1.set_xlim(-20,20)
                    ax1.set_ylim(-35,35)
                    ax1.set_zlim(-5,25)

                    ax2=fig.add_subplot(132, projection='3d')
                    ax2.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax2.scatter(Y_embed[indicesZ[0][i],0], Y_embed[indicesZ[0][i],1], Y_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
                    ax2.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
                    ax2.set_title('Y')
                    ax2.set_xlabel('Y(t)')
                    ax2.set_ylabel(f'Y(t-{tau})')
                    ax2.set_zlabel(f'Y(t-{2*tau})')
                    ax2.set_xlim(-30,30)
                    ax2.set_ylim(-40,40)
                    ax2.set_zlim(-5,40)

                    ax3=fig.add_subplot(133, projection='3d')
                    ax3.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
                    for i in range(n_neigh):
                        ax3.scatter(Z_embed[indicesZ[0][i],0], Z_embed[indicesZ[0][i],1], Z_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
                    ax3.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
                    ax3.set_title('Z')
                    ax3.set_xlabel('Z(t)')
                    ax3.set_ylabel(f'Z(t-{tau})')
                    ax3.set_zlabel(f'Z(t-{2*tau})')
                    ax3.set_xlim(-3,60)
                    ax3.set_ylim(-3,60)
                    ax3.set_zlim(-3,60)

                    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Z to map back to X and Y'+ '\nnoiseWhen: '+noiseWhen+'\nnoiseAddType: '+noiseAddType+'\nnoiseLevel: '+str(noiseLevel))
                    plt.savefig(os.path.join(case_save_dir,str(noiseLevel)+'Z2XY.png'))
                    plt.close()

                # create gif
                # imgs_X2YZ = [imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'X2YZ.png')) for noiseLevel in list_noiseLevel]
                # imgs_Y2XZ = [imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'Y2XZ.png')) for noiseLevel in list_noiseLevel]
                # imgs_Z2XY = [imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'Z2XY.png')) for noiseLevel in list_noiseLevel]
                imgs_X2YZ=[]
                imgs_Y2XZ=[]
                imgs_Z2XY=[]
                # append the no noise case first
                imgs_X2YZ.append(imageio.imread(os.path.join(output_dir,'None','NoneX2YZ.png')))
                imgs_Y2XZ.append(imageio.imread(os.path.join(output_dir,'None','NoneY2XZ.png')))
                imgs_Z2XY.append(imageio.imread(os.path.join(output_dir,'None','NoneZ2XY.png')))
                # append the noisy cases
                for noiseLevel in list_noiseLevel:
                    imgs_X2YZ.append(imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'X2YZ.png')))
                    imgs_Y2XZ.append(imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'Y2XZ.png')))
                    imgs_Z2XY.append(imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'Z2XY.png')))

                imageio.mimsave(os.path.join(gifs_dir,noiseType_str+'_'+noiseWhen+'_'+noiseAddType+'X2YZ.gif'), imgs_X2YZ, duration=700, loop=0)
                imageio.mimsave(os.path.join(gifs_dir,noiseType_str+'_'+noiseWhen+'_'+noiseAddType+'Y2XZ.gif'), imgs_Y2XZ, duration=700, loop=0)
                imageio.mimsave(os.path.join(gifs_dir,noiseType_str+'_'+noiseWhen+'_'+noiseAddType+'Z2XY.gif'), imgs_Z2XY, duration=700, loop=0)