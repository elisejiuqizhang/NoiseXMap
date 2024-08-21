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

data_source_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_noise')
data_source_filter_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_noise_filters')
viz_save_dir=os.path.join(root, 'outputs','viz', 'ccm_increasingL_noise_avg_downsample')

# list_noiseLevels=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
# list_systems=['Lorenz', 'RosslerLorenz']
# list_noiseTypes=['gNoise', 'lpNoise']
# list_noiseAddTypes=['add', 'mult', 'both']
# list_noiseWhen=['in', 'post']
# list_downsampleTypes=['average', 'decimation', 'subsample']
# list_downsampleFactors=[3,5,8,10]

list_noiseLevels=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
# list_noiseLevels=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

list_systems=['Lorenz']
# list_systems=['RosslerLorenz']

# list_noiseTypes=['gNoise', 'lpNoise']
list_noiseTypes=['lpNoise']
# list_noiseTypes=['gNoise']

# list_noiseAddTypes=['add', 'mult', 'both']
list_noiseAddTypes=['add']
# list_noiseAddTypes=['mult']
# list_noiseAddTypes=['both']

# list_noiseWhen=['in', 'post']
# list_noiseWhen=['in']
list_noiseWhen=['post']

# list_downsampleTypes=['average', 'decimation', 'subsample']
# list_downsampleTypes=['average', 'decimation']
list_downsampleTypes=['average']
# list_downsampleTypes=['subsample']

list_downsampleFactors=[3,5,8,10]
# list_downsampleFactors=[3,5]


# list_filters=["average", "median", "gaussian", "butterworth"]
list_filters=["average"]

# list_filterFactors=[5,8,10]
list_filterFactors=[5]

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
        noNoise_dir=os.path.join(data_source_dir, system+'('+ce_pair+')', 'noDownsample', f'tau{tau}_emd{emd}', 'noNoise')
        noNoise_arr_sc1=np.load(os.path.join(noNoise_dir, 'arr_sc1.npy'))
        noNoise_arr_sc2=np.load(os.path.join(noNoise_dir, 'arr_sc2.npy'))
        # assume the last column values are the converged highest value, take average and this will be the reference line in the plot
        noNoise_avg_sc1=np.mean(noNoise_arr_sc1[:,-1])
        noNoise_avg_sc2=np.mean(noNoise_arr_sc2[:,-1])

        # noise cases
        for noiseType in list_noiseTypes: # gNoise, lpNoise
            for noiseAddType in list_noiseAddTypes: # add, mult, both
                for noiseWhen in list_noiseWhen: # in, post
                    # noise data - no downsample directory
                    noise_noD_dir=os.path.join(data_source_dir, system+'('+ce_pair+')', 'noDownsample', f'tau{tau}_emd{emd}', noiseType, noiseWhen+'_'+noiseAddType+'_')
                    # load the .npy files in order of the noise levels
                    noise_noD_avg_sc1=[]
                    noise_noD_avg_sc2=[]
                    for noiseLevel in list_noiseLevels: # from 0.005, 0.01, 0.015, 0.02,..., 0.75
                        noise_noD_arr_sc1=np.load(os.path.join(noise_noD_dir+str(noiseLevel), 'arr_sc1.npy'))
                        noise_noD_arr_sc2=np.load(os.path.join(noise_noD_dir+str(noiseLevel), 'arr_sc2.npy'))
                        # noise_noD_avg_sc1.append(np.mean(noise_noD_arr_sc1[:,-1]))
                        # noise_noD_avg_sc2.append(np.mean(noise_noD_arr_sc2[:,-1]))
                        noise_noD_avg_sc1.append(np.mean(noise_noD_arr_sc1[:,-3:-1]))
                        noise_noD_avg_sc2.append(np.mean(noise_noD_arr_sc2[:,-3:-1]))

                    # noise data - downsampled direction
                    noise_D_dir=os.path.join(data_source_dir, system+'('+ce_pair+')', 'Downsampled')
                    for downsampleType in list_downsampleTypes: # average, decimation, subsample - each type is having its own plot along with the reference and the noise-noDownsample curves
                        noise_DType_dir=os.path.join(noise_D_dir, downsampleType+'_')
                        # each DType and DFactor gets their own curve plotted
                        list_noise_D_sc1_acrossDownFactors=[] 
                        list_noise_D_sc2_acrossDownFactors=[]
                        for downsampleFactor in list_downsampleFactors:
                            noise_D_factor_dir=os.path.join(noise_DType_dir+str(downsampleFactor),f'tau{tau}_emd{emd}', noiseType, noiseWhen+'_'+noiseAddType+'_')
                            # load the .npy files in order of the noise levels
                            noise_D_avg_sc1=[]
                            noise_D_avg_sc2=[]
                            for noiseLevel in list_noiseLevels: # from 0.005, 0.01, 0.015, 0.02,..., 0.75
                                noise_D_arr_sc1=np.load(os.path.join(noise_D_factor_dir+str(noiseLevel), 'arr_sc1.npy'))
                                noise_D_arr_sc2=np.load(os.path.join(noise_D_factor_dir+str(noiseLevel), 'arr_sc2.npy'))
                                # noise_D_avg_sc1.append(np.mean(noise_D_arr_sc1[:,-1]))
                                # noise_D_avg_sc2.append(np.mean(noise_D_arr_sc2[:,-1]))
                                noise_D_avg_sc1.append(np.mean(noise_D_arr_sc1[:,-3:-1]))
                                noise_D_avg_sc2.append(np.mean(noise_D_arr_sc2[:,-3:-1]))
                            list_noise_D_sc1_acrossDownFactors.append(noise_D_avg_sc1)
                            list_noise_D_sc2_acrossDownFactors.append(noise_D_avg_sc2)

                        # noise_data - only filters, no downsampling
                        for filterType in list_filters:
                            list_noise_filter_sc1_acrossFilterFactors=[]
                            list_noise_filter_sc2_acrossFilterFactors=[]
                            for filterFactor in list_filterFactors:
                                noise_filter_dir=os.path.join(data_source_filter_dir, system+'('+ce_pair+')', 'Filtered', filterType+'_'+str(filterFactor), f'tau{tau}_emd{emd}', noiseType, noiseWhen+'_'+noiseAddType+'_')
                                noise_filter_avg_sc1=[]
                                noise_filter_avg_sc2=[]
                                for noiseLevel in list_noiseLevels:
                                    noise_filter_arr_sc1=np.load(os.path.join(noise_filter_dir+str(noiseLevel), 'arr_sc1.npy'))
                                    noise_filter_arr_sc2=np.load(os.path.join(noise_filter_dir+str(noiseLevel), 'arr_sc2.npy'))
                                    noise_filter_avg_sc1.append(np.mean(noise_filter_arr_sc1[:,-3:-1]))
                                    noise_filter_avg_sc2.append(np.mean(noise_filter_arr_sc2[:,-3:-1]))
                                list_noise_filter_sc1_acrossFilterFactors.append(noise_filter_avg_sc1)
                                list_noise_filter_sc2_acrossFilterFactors.append(noise_filter_avg_sc2)

                        # plot the curves: for each case, plot the ref, noise-noD, noise-Ds (all difference D factors as one individual curve)
                        if not os.path.exists(os.path.join(viz_save_dir,system)):
                            os.makedirs(os.path.join(viz_save_dir,system))

                        # score 1 plot
                        fig=plt.figure(figsize=(13,8))
                        ax=fig.add_subplot(111)
                        ax.plot(list_noiseLevels, [noNoise_avg_sc1]*len(list_noiseLevels), 'k--', label='No noise')
                        ax.plot(list_noiseLevels, noise_noD_avg_sc1, color='c', label=f'{noiseType}_{noiseAddType}_{noiseWhen}-no Downsampling')
                        for i in range(len(list_downsampleFactors)):
                            ax.plot(list_noiseLevels, list_noise_D_sc1_acrossDownFactors[i], label=f'{noiseType}_{noiseAddType}_{noiseWhen}-{downsampleType}_{list_downsampleFactors[i]}')
                        for i in range(len(list_filterFactors)):
                            ax.plot(list_noiseLevels, list_noise_filter_sc1_acrossFilterFactors[i], label=f'{noiseType}_{noiseAddType}_{noiseWhen}-filter_{list_filters[i]}_{list_filterFactors[i]}')
                        ax.set_title(f'{system}_(Cause-Effect Pair:{ce_pair})-sc1')
                        ax.set_xlabel('Noise Levels')
                        ax.set_ylabel('Converged Correlation Score')
                        ax.legend()
                        plt.savefig(os.path.join(viz_save_dir,system, f'{ce_pair}_{noiseType}_{noiseAddType}_{noiseWhen}_{downsampleType}_sc1.png'))
                        plt.close()

                        # score 2 plot
                        fig=plt.figure(figsize=(13,8))
                        ax=fig.add_subplot(111)
                        ax.plot(list_noiseLevels, [noNoise_avg_sc2]*len(list_noiseLevels), 'k--', label='No noise')
                        ax.plot(list_noiseLevels, noise_noD_avg_sc2, color='c', label=f'{noiseType}_{noiseAddType}_{noiseWhen}-no Downsampling')
                        for i in range(len(list_downsampleFactors)):
                            ax.plot(list_noiseLevels, list_noise_D_sc2_acrossDownFactors[i], label=f'{noiseType}_{noiseAddType}_{noiseWhen}-{downsampleType}_{list_downsampleFactors[i]}')
                        for i in range(len(list_filterFactors)):
                            ax.plot(list_noiseLevels, list_noise_filter_sc2_acrossFilterFactors[i], label=f'{noiseType}_{noiseAddType}_{noiseWhen}-filter_{list_filters[i]}_{list_filterFactors[i]}')
                        ax.set_title(f'{system}_(Cause-Effect Pair:{ce_pair})-sc2')
                        ax.set_xlabel('Noise Levels')
                        ax.set_ylabel('Converged Correlation Score')
                        ax.legend()
                        plt.savefig(os.path.join(viz_save_dir,system, f'{ce_pair}_{noiseType}_{noiseAddType}_{noiseWhen}_{downsampleType}_sc2.png'))
                        plt.close()



                        



