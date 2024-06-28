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


import argparse
parser = argparse.ArgumentParser('Downsample noisy Lorenz and visualize neighborhoods in XMap')
parser.add_argument('--downsampleType', type=str, default=None, help='downsample type, options: None, "a/av/average" (average), "d/de/decimation" (remove/discard the rest), "s/sub/subsample" (randomly sample a subset of half the interval size from each interval, then average)')
parser.add_argument('--downsampleFactor', type=int, default=10, help='downsample interval')

parser.add_argument('--noiseType', type=str, default='lpNoise', help='noise type to use, options: None, "laplacian"/"lpNoise"/"l", "gaussian"/"gNoise"/"g"')
parser.add_argument('--noiseWhen', type=str, default='in', help='when to add noise, options: "in-generation"/"in", "post-generation"/"post", only effective when noiseType is not None')
parser.add_argument('--noiseAddType', type=str, default='add', help='additive or multiplicative noise, options: "additive"/"add", "multiplicative"/"mult", "both", only effective when noiseType is not None')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level, only effective when noiseType is not None')

parser.add_argument('--tau', type=int, default=2, help="CCM tau-lag")
# emd dim fixed at 3 for visualization, so no need to add as an input argument

parser.add_argument('--n_neigh', type=int, default=14, help='number of neighbors to consider in XMap')

parser.add_argument('--L', type=int, default=10000, help='length of the time series')
parser.add_argument('--plotL', type=int, default=10000, help='length of the time series to plot')

parser.add_argument('--seed', type=int, default=597, help='random seed for reproducibility')

args = parser.parse_args()

tau=args.tau
emd=3

# set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

# output dir
output_dir = os.path.join(root, 'outputs', 'viz', 'XMapDE', 'Lorenz')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# if no downsampling, save to subfolder "noDownsample"
if args.downsampleType==None or args.downsampleType.lower() == 'none':
    output_dir = os.path.join(output_dir, 'noDownsample')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    output_dir = os.path.join(output_dir, 'Downsampled', args.downsampleType+'_'+str(args.downsampleFactor))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

n_neigh=args.n_neigh
# L and plotL depending on whether there is downsampling
if args.downsampleType is None or args.downsampleType.lower() == 'none':
    L=args.L
    plotL=args.plotL
    start_plot=np.random.randint(0, L-plotL) if L>plotL else 0
    centroid_idx=np.random.randint(0, plotL-1)
else:
    L=args.L//args.downsampleFactor
    plotL=args.plotL//args.downsampleFactor
    start_plot=np.random.randint(0, L-plotL) if L>plotL else 0
    centroid_idx=np.random.randint(0, plotL-1)

noiseType_str=args.noiseType if (args.noiseType!=None and args.noiseType.lower()!='none') else 'None'
imgs_dir = os.path.join(output_dir, noiseType_str)
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

if args.noiseType!=None and args.noiseType.lower()!='none':
    case_save_dir=os.path.join(imgs_dir,args.noiseWhen,args.noiseAddType)
    if not os.path.exists(case_save_dir):
        os.makedirs(case_save_dir)
    # generate data
    data=gen_Lorenz(noiseType=args.noiseType, noiseWhen=args.noiseWhen, noiseAddType=args.noiseAddType, noiseLevel=args.noiseLevel, L=args.L)
else:
    case_save_dir=imgs_dir
    # generate data
    data=gen_Lorenz(noiseType=None, L=args.L)

# downsample
if args.downsampleType!=None and args.downsampleType.lower()!='none':
    if args.downsampleType.lower() in ['a', 'av', 'average']:
        data_downsampled=np.zeros((args.L//args.downsampleFactor, 3))
        for i in range(0, args.L, args.downsampleFactor):
            data_downsampled[i//args.downsampleFactor]=np.mean(data[i:i+args.downsampleFactor], axis=0)
    elif args.downsampleType.lower() in ['d', 'de', 'decimation']:
        data_downsampled=data[::args.downsampleFactor]
    elif args.downsampleType.lower() in ['s', 'sub', 'subsample']:
        # my way of subsampling: 
        # In each segment of length args.downsampleFactor, randomly sample half of the interval size within, then take average
        data_downsampled=np.zeros((args.L//args.downsampleFactor,3))
        for i in range(0, args.L, args.downsampleFactor):
            rdm_start=np.random.randint(0, args.downsampleFactor//2)
            data_downsampled[i//args.downsampleFactor]=np.mean(data[i+rdm_start:i+rdm_start+args.downsampleFactor//2], axis=0)
    else:
        raise ValueError('downsampleType not recognized')
else:
    data_downsampled=data

# plot the data
data_downsampled_plot=data_downsampled[start_plot:start_plot+plotL]
data_X=data_downsampled_plot[:,0]
data_Y=data_downsampled_plot[:,1]
data_Z=data_downsampled_plot[:,2]
X_embed=time_delay_embed(data_X, tau, emd)
Y_embed=time_delay_embed(data_Y, tau, emd)
Z_embed=time_delay_embed(data_Z, tau, emd)


neigh = NearestNeighbors(n_neighbors=n_neigh)
# find k nearest neighbors on the real data
neigh.fit(data_downsampled_plot)
_, indices = neigh.kneighbors(data_downsampled_plot[centroid_idx].reshape(1, -1))
# find k nearest neighbors on X_embed
neigh.fit(X_embed)
_, indicesX = neigh.kneighbors(X_embed[centroid_idx].reshape(1, -1))
# find k nearest neighbors on Y_embed
neigh.fit(Y_embed)
_, indicesY = neigh.kneighbors(Y_embed[centroid_idx].reshape(1, -1))
# find k nearest neighbors on Z_embed
neigh.fit(Z_embed)
_, indicesZ = neigh.kneighbors(Z_embed[centroid_idx].reshape(1, -1))

# 3D plot: four imgs side by side, from left to right - real data, X_embed, Y_embed, Z_embed
# 1. using indices to plot
fig=plt.figure(figsize=(36,9))
ax1=fig.add_subplot(141, projection='3d')
ax1.plot(data_X, data_Y, data_Z, color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax1.scatter(data_X[indices[0][i]], data_Y[indices[0][i]], data_Z[indices[0][i]], c='g', alpha=0.5, s=10)
ax1.scatter(data_X[centroid_idx], data_Y[centroid_idx], data_Z[centroid_idx], c='r', s=18)
ax1.set_title('Real Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim([-35, 35])
ax1.set_ylim([-30, 30])
ax1.set_zlim([-3, 40])

ax2=fig.add_subplot(142, projection='3d')
ax2.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax2.scatter(X_embed[indices[0][i],0], X_embed[indices[0][i],1], X_embed[indices[0][i],2], c='g', alpha=0.5, s=10)
ax2.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
ax2.set_title('X_embed')
ax2.set_xlabel('X(t)')
ax2.set_ylabel('X(t-tau)')
ax2.set_zlabel('X(t-2*tau)')
ax2.set_xlim(-15,15)
ax2.set_ylim(-40,40)
ax2.set_zlim(-5,20)

ax3=fig.add_subplot(143, projection='3d')
ax3.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax3.scatter(Y_embed[indices[0][i],0], Y_embed[indices[0][i],1], Y_embed[indices[0][i],2], c='g', alpha=0.5, s=10)
ax3.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
ax3.set_title('Y_embed')
ax3.set_xlabel('Y(t)')
ax3.set_ylabel('Y(t-tau)')
ax3.set_zlabel('Y(t-2*tau)')
ax3.set_xlim(-20,20)
ax3.set_ylim(-45,45)
ax3.set_zlim(-5,30)

ax4=fig.add_subplot(144, projection='3d')
ax4.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax4.scatter(Z_embed[indices[0][i],0], Z_embed[indices[0][i],1], Z_embed[indices[0][i],2], c='g', alpha=0.5, s=10)
ax4.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
ax4.set_title('Z_embed')
ax4.set_xlabel('Z(t)')
ax4.set_ylabel('Z(t-tau)')
ax4.set_zlabel('Z(t-2*tau)')
ax4.set_xlim(-3,50)
ax4.set_ylim(-3,50)
ax4.set_zlim(-3,50)

# super title (check noiseType)
if args.noiseType!=None and args.noiseType.lower()!='none':
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing real data indices to map back to delay embeddings of X, Y and Z'+'\nnoiseWhen: '+args.noiseWhen+'\nnoiseAddType: '+args.noiseAddType+'\nnoiseLevel: '+str(args.noiseLevel))
    plt.savefig(os.path.join(case_save_dir,str(args.noiseLevel)+'Real2XYZ.png'))
else:
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing real data indices to map back to delay embeddings of X, Y and Z')
    plt.savefig(os.path.join(case_save_dir,'Real2XYZ.png'))
plt.close()

# 2. using indicesX to plot
fig=plt.figure(figsize=(36,9))
ax1=fig.add_subplot(141, projection='3d')
ax1.plot(data_X, data_Y, data_Z, color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax1.scatter(data_X[indicesX[0][i]], data_Y[indicesX[0][i]], data_Z[indicesX[0][i]], c='g', alpha=0.5, s=10)
ax1.scatter(data_X[centroid_idx], data_Y[centroid_idx], data_Z[centroid_idx], c='r', s=18)
ax1.set_title('Real Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim([-35, 35])
ax1.set_ylim([-30, 30])
ax1.set_zlim([-3, 40])

ax2=fig.add_subplot(142, projection='3d')
ax2.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax2.scatter(X_embed[indicesX[0][i],0], X_embed[indicesX[0][i],1], X_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
ax2.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
ax2.set_title('X_embed')
ax2.set_xlabel('X(t)')
ax2.set_ylabel('X(t-tau)')
ax2.set_zlabel('X(t-2*tau)')
ax2.set_xlim(-15,15)
ax2.set_ylim(-40,40)
ax2.set_zlim(-5,20)

ax3=fig.add_subplot(143, projection='3d')
ax3.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax3.scatter(Y_embed[indicesX[0][i],0], Y_embed[indicesX[0][i],1], Y_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
ax3.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
ax3.set_title('Y_embed')
ax3.set_xlabel('Y(t)')
ax3.set_ylabel('Y(t-tau)')
ax3.set_zlabel('Y(t-2*tau)')
ax3.set_xlim(-20,20)
ax3.set_ylim(-45,45)
ax3.set_zlim(-5,30)

ax4=fig.add_subplot(144, projection='3d')
ax4.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax4.scatter(Z_embed[indicesX[0][i],0], Z_embed[indicesX[0][i],1], Z_embed[indicesX[0][i],2], c='g', alpha=0.5, s=10)
ax4.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
ax4.set_title('Z_embed')
ax4.set_xlabel('Z(t)')
ax4.set_ylabel('Z(t-tau)')
ax4.set_zlabel('Z(t-2*tau)')
ax4.set_xlim(-3,50)
ax4.set_ylim(-3,50)
ax4.set_zlim(-3,50)

# super title (check noiseType)
if args.noiseType!=None and args.noiseType.lower()!='none':
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing X_embed indices to map back to delay embeddings of X, Y and Z'+'\nnoiseWhen: '+args.noiseWhen+'\nnoiseAddType: '+args.noiseAddType+'\nnoiseLevel: '+str(args.noiseLevel))
    plt.savefig(os.path.join(case_save_dir,str(args.noiseLevel)+'X2RealYZ.png'))
else:
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing X_embed indices to map back to delay embeddings of X, Y and Z')
    plt.savefig(os.path.join(case_save_dir,'X2RealYZ.png'))
plt.close()

# 3. using indicesY to plot
fig=plt.figure(figsize=(36,9))
ax1=fig.add_subplot(141, projection='3d')
ax1.plot(data_X, data_Y, data_Z, color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax1.scatter(data_X[indicesY[0][i]], data_Y[indicesY[0][i]], data_Z[indicesY[0][i]], c='g', alpha=0.5, s=10)
ax1.scatter(data_X[centroid_idx], data_Y[centroid_idx], data_Z[centroid_idx], c='r', s=18)
ax1.set_title('Real Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim([-35, 35])
ax1.set_ylim([-30, 30])
ax1.set_zlim([-3, 40])

ax2=fig.add_subplot(142, projection='3d')
ax2.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax2.scatter(X_embed[indicesY[0][i],0], X_embed[indicesY[0][i],1], X_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
ax2.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
ax2.set_title('X_embed')
ax2.set_xlabel('X(t)')
ax2.set_ylabel('X(t-tau)')
ax2.set_zlabel('X(t-2*tau)')
ax2.set_xlim(-15,15)
ax2.set_ylim(-40,40)
ax2.set_zlim(-5,20)

ax3=fig.add_subplot(143, projection='3d')
ax3.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax3.scatter(Y_embed[indicesY[0][i],0], Y_embed[indicesY[0][i],1], Y_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
ax3.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
ax3.set_title('Y_embed')
ax3.set_xlabel('Y(t)')
ax3.set_ylabel('Y(t-tau)')
ax3.set_zlabel('Y(t-2*tau)')
ax3.set_xlim(-20,20)
ax3.set_ylim(-45,45)
ax3.set_zlim(-5,30)

ax4=fig.add_subplot(144, projection='3d')
ax4.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax4.scatter(Z_embed[indicesY[0][i],0], Z_embed[indicesY[0][i],1], Z_embed[indicesY[0][i],2], c='g', alpha=0.5, s=10)
ax4.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
ax4.set_title('Z_embed')
ax4.set_xlabel('Z(t)')
ax4.set_ylabel('Z(t-tau)')
ax4.set_zlabel('Z(t-2*tau)')
ax4.set_xlim(-3,50)
ax4.set_ylim(-3,50)
ax4.set_zlim(-3,50)

# super title (check noiseType)
if args.noiseType!=None and args.noiseType.lower()!='none':
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Y_embed indices to map back to delay embeddings of X, Y and Z'+'\nnoiseWhen: '+args.noiseWhen+'\nnoiseAddType: '+args.noiseAddType+'\nnoiseLevel: '+str(args.noiseLevel))
    plt.savefig(os.path.join(case_save_dir,str(args.noiseLevel)+'Y2RealXZ.png'))
else:
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Y_embed indices to map back to delay embeddings of X, Y and Z')
    plt.savefig(os.path.join(case_save_dir,'Y2RealXZ.png'))
plt.close()

# 4. using indicesZ to plot
fig=plt.figure(figsize=(36,9))
ax1=fig.add_subplot(141, projection='3d')
ax1.plot(data_X, data_Y, data_Z, color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax1.scatter(data_X[indicesZ[0][i]], data_Y[indicesZ[0][i]], data_Z[indicesZ[0][i]], c='g', alpha=0.5, s=10)
ax1.scatter(data_X[centroid_idx], data_Y[centroid_idx], data_Z[centroid_idx], c='r', s=18)
ax1.set_title('Real Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim([-35, 35])
ax1.set_ylim([-30, 30])
ax1.set_zlim([-3, 40])

ax2=fig.add_subplot(142, projection='3d')
ax2.plot(X_embed[:,0], X_embed[:,1], X_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax2.scatter(X_embed[indicesZ[0][i],0], X_embed[indicesZ[0][i],1], X_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
ax2.scatter(X_embed[centroid_idx,0], X_embed[centroid_idx,1], X_embed[centroid_idx,2], c='r', s=18)
ax2.set_title('X_embed')
ax2.set_xlabel('X(t)')
ax2.set_ylabel('X(t-tau)')
ax2.set_zlabel('X(t-2*tau)')
ax2.set_xlim(-15,15)
ax2.set_ylim(-40,40)
ax2.set_zlim(-5,20)

ax3=fig.add_subplot(143, projection='3d')
ax3.plot(Y_embed[:,0], Y_embed[:,1], Y_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax3.scatter(Y_embed[indicesZ[0][i],0], Y_embed[indicesZ[0][i],1], Y_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
ax3.scatter(Y_embed[centroid_idx,0], Y_embed[centroid_idx,1], Y_embed[centroid_idx,2], c='r', s=18)
ax3.set_title('Y_embed')
ax3.set_xlabel('Y(t)')
ax3.set_ylabel('Y(t-tau)')
ax3.set_zlabel('Y(t-2*tau)')
ax3.set_xlim(-20,20)
ax3.set_ylim(-45,45)
ax3.set_zlim(-5,30)

ax4=fig.add_subplot(144, projection='3d')
ax4.plot(Z_embed[:,0], Z_embed[:,1], Z_embed[:,2], color='blue', linewidth=0.5,alpha=0.45)
for i in range(n_neigh):
    ax4.scatter(Z_embed[indicesZ[0][i],0], Z_embed[indicesZ[0][i],1], Z_embed[indicesZ[0][i],2], c='g', alpha=0.5, s=10)
ax4.scatter(Z_embed[centroid_idx,0], Z_embed[centroid_idx,1], Z_embed[centroid_idx,2], c='r', s=18)
ax4.set_title('Z_embed')
ax4.set_xlabel('Z(t)')
ax4.set_ylabel('Z(t-tau)')
ax4.set_zlabel('Z(t-2*tau)')
ax4.set_xlim(-3,50)
ax4.set_ylim(-3,50)
ax4.set_zlim(-3,50)

# super title (check noiseType)
if args.noiseType!=None and args.noiseType.lower()!='none':
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Z_embed indices to map back to delay embeddings of X, Y and Z'+'\nnoiseWhen: '+args.noiseWhen+'\nnoiseAddType: '+args.noiseAddType+'\nnoiseLevel: '+str(args.noiseLevel))
    plt.savefig(os.path.join(case_save_dir,str(args.noiseLevel)+'Z2RealXY.png'))
else:
    fig.suptitle('Lorenz attractor - noiseType: '+noiseType_str+'\nUsing Z_embed indices to map back to delay embeddings of X, Y and Z')
    plt.savefig(os.path.join(case_save_dir,'Z2RealXY.png'))
plt.close()