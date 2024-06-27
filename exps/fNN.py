# use fNN to determine optimal embedding dimension
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from utils.XMap.fNN import fNN

import argparse
parser=argparse.ArgumentParser('fNN for optimal embedding dimension with varying noises')

parser.add_argument('--dataType', type=str, default='BiVarLogistic', help='data type to use, options: "BiVarLogistic" ("BiLog"), "Lorenz" ("L"), "RosslerLorenz" ("RL")')
parser.add_argument('--noiseType', type=str, default=None, help='noise type to use, options: None, "laplacian"/"lpNoise"/"l", "gaussian"/"gNoise"/"g"')
parser.add_argument('--noiseWhen', type=str, default='in', help='when to add noise, options: "in-generation"/"in", "post-generation"/"post", only effective when noiseType is not None')
parser.add_argument('--noiseAddType', type=str, default='add', help='additive or multiplicative noise, options: "additive"/"add", "multiplicative"/"mult", "both", only effective when noiseType is not None')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level, only effective when noiseType is not None')

parser.add_argument('--tau', type=int, default=1, help="tau-lag")
parser.add_argument('--emd_min', type=int, default=2, help="embedding dimension - min")
parser.add_argument('--emd_max', type=int, default=12, help="embedding dimension - max")
parser.add_argument('--emd_step', type=int, default=1, help="embedding dimension - step")   

parser.add_argument('--knn', type=int, default=10, help="Neighborhood size")
parser.add_argument('--L', type=int, default=1000, help="Length of input time series for the test")
parser.add_argument('--num_trials', type=int, default=10, help="Number of trials for the test")

parser.add_argument('--seed', type=int, default=97, help="random seed to load the data at a random start point")

args=parser.parse_args()

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)


# save path
output_dir=os.path.join(root, 'outputs', 'exps', 'fNN_optimE')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# read data, since it is already generated
data_dir=os.path.join(root, 'data', args.dataType)
if args.noiseType==None or args.noiseType.lower()=='none':
    file_name=os.path.join(data_dir, 'noNoise.csv')
else:
    file_name=os.path.join(data_dir, args.noiseType+"_"+args.noiseWhen+"_"+args.noiseAddType+"_"+str(args.noiseLevel)+'.csv')
    
df=pd.read_csv(file_name)

if args.dataType.lower()=='bivarlogistic' or args.dataType.lower()=='bilog':
    var_list=['X', 'Y']
elif args.dataType.lower()=='lorenz' or args.dataType.lower()=='l':
    var_list=['X', 'Y', 'Z']
elif args.dataType.lower()=='rosslerlorenz' or args.dataType.lower()=='rl':
    var_list=['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']

totalL=len(df)


# save path - add subfolder for the data type
output_dir=os.path.join(output_dir, args.dataType)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save path - add tau, kNN
output_dir=os.path.join(output_dir, 'tau'+str(args.tau), 'kNN'+str(args.knn))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save path - add subfolder for the noise type
if args.noiseType==None or args.noiseType.lower()=='none':
    output_dir=os.path.join(output_dir, 'noNoise')
else:
    output_dir=os.path.join(output_dir, args.noiseType, args.noiseWhen+"_"+args.noiseAddType+"_"+str(args.noiseLevel))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


arr_E=np.zeros((args.num_trials, len(var_list)))
# generate args.num_trials number of start points
start_points=np.random.randint(0, totalL-args.L-args.emd_max*args.tau-1, args.num_trials)

for idx in range(args.num_trials):
    start_point=start_points[idx]
    print('Trial:', idx, 'Start point:', start_point)
    for var in var_list:
        print('Variable name:', var)
        ts=df[var].values[start_point:start_point+args.L+args.emd_max*args.tau]
        # use fNN to determine optimal embedding dimension
        boo, emd, ratio = fNN(ts, tau=args.tau, emd_min=args.emd_min, emd_max=args.emd_max, emd_step=args.emd_step, knn=args.knn, L=args.L)
        arr_E[idx, var_list.index(var)]=emd
        print('Optimal embedding dimension:', emd, 'False nearest neighbor ratio:', ratio,'\n\n')
        with open(output_dir+'/'+str(idx)+'_'+var+'.txt', 'w') as f:
            f.write('Optimal embedding dimension: '+str(emd)+'\nFalse nearest neighbor ratio: '+str(ratio)+'\n')

# save the results
# save as csv
df_E=pd.DataFrame(arr_E, columns=var_list)
df_E.to_csv(output_dir+'/optimal_emd.csv', index=False)
