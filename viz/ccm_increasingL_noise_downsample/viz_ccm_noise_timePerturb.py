# read outputs and plot the three curves: 
# 1. converged correlation values of no noise and no downsampling case (reference)
# 2. converged correlation values of with noise and no downsampling case (expected to be lowest)
# 3. converged correlation values of with noise and downsampling case (expected to be higher than 2)

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root=os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(root)

data_source_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_perturbation')
# data_source_filter_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_noise_filters')
viz_save_dir=os.path.join(root, 'outputs','viz', 'ccm_increasingL_perturbation_downsample')

# list_noiseLevels=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
# list_systems=['Lorenz', 'RosslerLorenz']
# list_noiseTypes=['gNoise', 'lpNoise']
# list_noiseAddTypes=['add', 'mult', 'both']
# list_noiseWhen=['in', 'post']
# list_downsampleTypes=['average', 'decimation', 'subsample']
# list_downsampleFactors=[3,5,8,10]



list_systems=['Lorenz']

list_lenInterv=[5,10]
list_percentInterv=[0.1,0.3,0.5]
list_swapPercent=[0.1,0.3,0.5,0.8]


list_downsampleTypes=['average', 'decimation', 'subsample']
# list_downsampleTypes=['average', 'decimation']
# list_downsampleTypes=['average']
# list_downsampleTypes=['subsample']

list_downsampleFactors=[5,8,10]
# list_downsampleFactors=[3,5]


# # list_filters=["average", "median", "gaussian", "butterworth"]
# list_filters=["average"]

# # list_filterFactors=[5,8,10]
# list_filterFactors=[5]

tau=1
emd=3

for system in list_systems:
    if system=='Lorenz':
        list_cause_effect_pairs=['XY','XZ','YZ']
        # list_cause_effect_pairs=['XY']
        # list_cause_effect_pairs=['XZ']
        # list_cause_effect_pairs=['YZ']
    elif system=='RosslerLorenz':
        # list_cause_effect_pairs=['X1X2','X1Y2','X1Z2','Y1X2','Y1Y2','Y1Z2','Z1X2','Z1Y2','Z1Z2']
        list_cause_effect_pairs=['X1X2']
    else:
        raise ValueError('Unknown system')
    
    for ce_pair in list_cause_effect_pairs:
        # 1. load the reference data (the no noise, no downsampling case)- will be a flat dashed line as reference in the plot
        # noNoise_dir=os.path.join(data_source_dir, system+'('+ce_pair+')', 'noDownsample', f'tau{tau}_emd{emd}', 'noNoise')
        data_noNoise_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_noise')
        noNoise_dir=os.path.join(data_noNoise_dir, system+'('+ce_pair+')', 'noDownsample', f'tau{tau}_emd{emd}', 'noNoise')
        noNoise_arr_sc1=np.load(os.path.join(noNoise_dir, 'arr_sc1.npy'))
        noNoise_arr_sc2=np.load(os.path.join(noNoise_dir, 'arr_sc2.npy'))
        # assume the last column values are the converged highest value, take average and this will be the reference line in the plot
        noNoise_avg_sc1=np.mean(noNoise_arr_sc1[:,-1])
        noNoise_avg_sc2=np.mean(noNoise_arr_sc2[:,-1])

        # perturbed cases
        for lenInterv in list_lenInterv: 
            for percentInterv in list_percentInterv: 
                for swapPercent in list_swapPercent: 
                    # perturbed: no downsampling
                    perturbed_noD_dir=os.path.join(data_source_dir, system+'('+ce_pair+')', 'noDownsample', f'tau{tau}_emd{emd}','withPerturb',f"lenInterv{lenInterv}_percentInterv{percentInterv}_swapPercent")
                    # load the .npy files in order of swapPercent
                    perturbed_noD_avg_sc1=[]
                    perturbed_noD_avg_sc2=[]
                    for swapPercent in list_swapPercent:
                        perturbed_noD_arr_sc1=np.load(os.path.join(perturbed_noD_dir+str(swapPercent), f'arr_sc1.npy'))
                        perturbed_noD_arr_sc2=np.load(os.path.join(perturbed_noD_dir+str(swapPercent), f'arr_sc2.npy'))
                        perturbed_noD_avg_sc1.append(np.mean(perturbed_noD_arr_sc1[:,-1]))
                        perturbed_noD_avg_sc2.append(np.mean(perturbed_noD_arr_sc2[:,-1]))


                    # perturbed: downsampling
                    perturbed_D_dir=os.path.join(data_source_dir, system+'('+ce_pair+')', 'Downsampled')
                    for downsampleType in list_downsampleTypes: # average, decimation, subsample - each type is having its own plot along with the reference and the noise-noDownsample curves
                        perturbed_DType_dir=os.path.join(perturbed_D_dir, downsampleType+'_')
                        # each DType and DFactor gets their own curve plotted
                        list_noise_D_sc1_acrossDownFactors=[] 
                        list_noise_D_sc2_acrossDownFactors=[]
                        for downsampleFactor in list_downsampleFactors:
                            perturbed_D_factor_dir=os.path.join(perturbed_DType_dir+str(downsampleFactor),f'tau{tau}_emd{emd}','withPerturb',f"lenInterv{lenInterv}_percentInterv{percentInterv}_swapPercent")                        
                            # load the .npy files in order of swapPercent
                            perturbed_D_avg_sc1=[]
                            perturbed_D_avg_sc2=[]
                            for swapPercent in list_swapPercent:
                                perturbed_D_arr_sc1=np.load(os.path.join(perturbed_D_factor_dir+str(swapPercent), f'arr_sc1.npy'))
                                perturbed_D_arr_sc2=np.load(os.path.join(perturbed_D_factor_dir+str(swapPercent), f'arr_sc2.npy'))
                                perturbed_D_avg_sc1.append(np.mean(perturbed_D_arr_sc1[:,-1]))
                                perturbed_D_avg_sc2.append(np.mean(perturbed_D_arr_sc2[:,-1]))

                            list_noise_D_sc1_acrossDownFactors.append(perturbed_D_avg_sc1)
                            list_noise_D_sc2_acrossDownFactors.append(perturbed_D_avg_sc2)

                        # plot the curves: for each case, plot the ref, noise-noD, noise-Ds (all difference D factors as one individual curve)
                        if not os.path.exists(os.path.join(viz_save_dir,system)):
                            os.makedirs(os.path.join(viz_save_dir,system))

                        # score 1 plot
                        fig=plt.figure(figsize=(13,8))
                        ax=fig.add_subplot(111)
                        ax.plot(list_swapPercent, [noNoise_avg_sc1]*len(list_swapPercent), 'k--', label='No noise')
                        ax.plot(list_swapPercent, perturbed_noD_avg_sc1, 'r-', label='Perturbed, no downsampling')
                        for i, downsampleFactor in enumerate(list_downsampleFactors):
                            ax.plot(list_swapPercent, list_noise_D_sc1_acrossDownFactors[i], label=f'Perturbed, {downsampleType}_{downsampleFactor}')
                        ax.set_xlabel('Swap percent')
                        ax.set_ylabel('sc1')
                        ax.set_title(f'{system}_(Cause-Effect Pair:{ce_pair})-sc1')
                        ax.legend()
                        fig.savefig(os.path.join(viz_save_dir,system,f'{ce_pair}_sc1_lenInterv{lenInterv}_percentInterv{percentInterv}_{downsampleType}.png'))
                        plt.close(fig)

                        # score 2 plot
                        fig=plt.figure(figsize=(13,8))
                        ax=fig.add_subplot(111)
                        ax.plot(list_swapPercent, [noNoise_avg_sc2]*len(list_swapPercent), 'k--', label='No noise')
                        ax.plot(list_swapPercent, perturbed_noD_avg_sc2, 'r-', label='Perturbed, no downsampling')
                        for i, downsampleFactor in enumerate(list_downsampleFactors):
                            ax.plot(list_swapPercent, list_noise_D_sc2_acrossDownFactors[i], label=f'Perturbed, {downsampleType}_{downsampleFactor}')
                        ax.set_xlabel('Swap percent')
                        ax.set_ylabel('sc2')
                        ax.set_title(f'{system}_(Cause-Effect Pair:{ce_pair})-sc2')
                        ax.legend()
                        fig.savefig(os.path.join(viz_save_dir,system,f'{ce_pair}_sc2_lenInterv{lenInterv}_percentInterv{percentInterv}_{downsampleType}.png'))
                        plt.close(fig)
                        



