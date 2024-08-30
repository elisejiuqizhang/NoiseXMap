# plot the changes of sc1 and sc2 with increasing filtering size

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

data_source_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_noise')
data_source_filter_dir=os.path.join(root, 'outputs','exps','ccm_increasingL_noise_filters')
viz_save_dir=os.path.join(root, 'outputs','viz', 'ccm_increasingL_noise_avg_only')

# list_filterFactors=[3,5,7,9,11,13]
list_filterFactors=[0,3,5,8,10]
filterType='average'

tau=1
emd=4

cause='rad'
effect='T_2m'

list_sc1_asFilterFactorIncrease=[]
list_sc2_asFilterFactorIncrease=[]

# 1. reference: no downsampling case
ref_data_dir=os.path.join(data_source_dir, 'ERA5('+cause+effect+')', 'noDownsample', f'tau{tau}_emd{emd}')
arr_sc1=np.load(os.path.join(ref_data_dir, 'arr_sc1.npy'))
arr_sc2=np.load(os.path.join(ref_data_dir, 'arr_sc2.npy'))
list_sc1_asFilterFactorIncrease.append(np.mean(arr_sc1[:,-2:]))
list_sc2_asFilterFactorIncrease.append(np.mean(arr_sc2[:,-2:]))
# 2. filtering case
for filterFactor in list_filterFactors:
    if filterFactor==0:
        continue
    else:
        filter_data_dir=os.path.join(data_source_filter_dir, 'ERA5('+cause+effect+')','Filtered', filterType+'_'+f'{filterFactor}', f'tau{tau}_emd{emd}')
        arr_sc1=np.load(os.path.join(filter_data_dir, 'arr_sc1.npy'))
        arr_sc2=np.load(os.path.join(filter_data_dir, 'arr_sc2.npy'))
        list_sc1_asFilterFactorIncrease.append(np.mean(arr_sc1[:,-2:]))
        list_sc2_asFilterFactorIncrease.append(np.mean(arr_sc2[:,-2:]))

plt.figure(figsize=(10,6))
plt.plot(list_filterFactors, list_sc1_asFilterFactorIncrease, label='sc1: '+cause+' -> '+effect)
plt.plot(list_filterFactors, list_sc2_asFilterFactorIncrease, label='sc2: '+cause+' <- '+effect)
plt.xlabel('Filtering size')
plt.ylabel('Converged correlation score')
plt.title(f'{cause} -> {effect}')
plt.legend()
plt.savefig(os.path.join(viz_save_dir, f'{cause} -> {effect}_tau{tau}_emd{emd}.png'))
plt.show()