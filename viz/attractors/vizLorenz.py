import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)


from utils.data_gen.Lorenz import *

import numpy as np
import random
import matplotlib.pyplot as plt
import imageio # generate gif that can loop out of the pngs

# set random seed for reproducibility
random.seed(97)
np.random.seed(97)

output_dir = os.path.join(root, 'outputs', 'viz', 'attractors', 'Lorenz')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

L = 10000
plot_L = 3000

# visualize in 3D the Lorenz attractor
list_noiseType=[None, 'Gaussian', 'Laplacian']
list_noiseWhen=['in', 'post']
list_noiseAddType=['add', 'mult', 'both']   
list_noiseLevel=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for noiseType in list_noiseType:
    if noiseType==None or noiseType.lower()=='none':
        noiseType_str = 'None'
        imgs_dir = os.path.join(output_dir, noiseType_str)
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        # generate non-noisy data
        data=gen_Lorenz(noiseType=noiseType, L=L)
        # 3D plot - no need to generate gif in this case
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:plot_L,0], data[:plot_L,1], data[:plot_L,2])
        plt.title('Lorenz attractor - noiseType: '+noiseType_str)
        plt.savefig(os.path.join(imgs_dir,noiseType_str+'.png'))
        plt.close()

    else: # noise cases
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
                    case_save_dir = os.path.join(imgs_dir,noiseType_str,noiseWhen,noiseAddType)
                    if not os.path.exists(case_save_dir):
                        os.makedirs(case_save_dir)
                    data=gen_Lorenz(noiseType=noiseType, noiseWhen=noiseWhen, noiseAddType=noiseAddType, noiseLevel=noiseLevel, L=L)
                    # 3D plot 
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # fix the range of three axes for better gif visualization results
                    ax.set_xlim(-50,50)
                    ax.set_ylim(-50,50)
                    ax.set_zlim(0,80)
                    ax.plot(data[:plot_L,0], data[:plot_L,1], data[:plot_L,2])
                    ax.x_label = 'x'
                    ax.y_label = 'y'
                    ax.z_label = 'z'

                    plt.title('noiseType: '+noiseType_str+'\nnoiseWhen: '+noiseWhen+'\nnoiseAddType: '+noiseAddType+'\nnoiseLevel: '+str(noiseLevel))
                    plt.savefig(os.path.join(case_save_dir,str(noiseLevel)+'.png'))
                    plt.close()

                # generate gif that loops forever
                images = [imageio.imread(os.path.join(case_save_dir,str(noiseLevel)+'.png')) for noiseLevel in list_noiseLevel]
                imageio.mimsave(os.path.join(gifs_dir,noiseType_str+'_'+noiseWhen+'_'+noiseAddType+'.gif'), images, duration=400, loop=0)