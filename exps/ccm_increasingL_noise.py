# comparing the CCM fit under no noise and with noise cases (no filtering yet)

import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from utils.data_gen.BiVarLogistic import gen_BiVarLogistic
from utils.data_gen.Lorenz import gen_Lorenz
from utils.data_gen.RosslerLorenz import gen_RosslerLorenz
from utils.XMap.CM_simplex import CM_simplex

import argparse
parser=argparse.ArgumentParser('viz increasing L vs. avgCorr for CCM')
parser.add_argument('--downsampleType', type=str, default='average', help='downsample type, options: None, "a/av/average" (average), "d/de/decimation" (remove/discard the rest), "s/sub/subsample" (randomly sample a subset of half the interval size from each interval, then average)')
parser.add_argument('--downsampleFactor', type=int, default=10, help='downsample interval')

parser.add_argument('--dataType', type=str, default='BiVarLogistic', help='data type to use, options: "BiVarLogistic" ("BiLog"), "Lorenz" ("L"), "RosslerLorenz" ("RL")')
parser.add_argument('--noiseType', type=str, default=None, help='noise type to use, options: None, "laplacian"/"lpNoise"/"l", "gaussian"/"gNoise"/"g"')
parser.add_argument('--noiseWhen', type=str, default='in', help='when to add noise, options: "in-generation"/"in", "post-generation"/"post", only effective when noiseType is not None')
parser.add_argument('--noiseAddType', type=str, default='add', help='additive or multiplicative noise, options: "additive"/"add", "multiplicative"/"mult", "both", only effective when noiseType is not None')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level, only effective when noiseType is not None')

parser.add_argument('--tau', type=int, default=1, help="CCM tau-lag")
parser.add_argument('--emd', type=int, default=2, help="CCM embedding dimension")

parser.add_argument('--maxL', type=int, default=10000, help='max L to test')
parser.add_argument('--numL', type=int, default=10, help='number of Ls to test, we will divide the range from 500 to maxL into numL parts')
parser.add_argument('--numSeeds', type=int, default=6, help='number of seeds to test, we will take average and std across all the seeds')

args=parser.parse_args()

# save path
output_dir=os.path.join(root, 'outputs', 'exps', 'ccm_increasingL_noise')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# # generate data - length need to be long enough to avoid transient and to enable randomly sample start points
# totalL=args.maxL+1000
# if args.dataType.lower()=='bivarlogistic' or args.dataType.lower()=='bilog':
#     data=gen_BiVarLogistic(a_x=3.7, a_y=3.72, b_xy=0.35, b_yx=0.06, noiseType=args.noiseType, noiseWhen=args.noiseWhen, noiseAddType=args.noiseAddType, noiseLevel=args.noiseLevel, L=totalL)
#     # wrap as pandas dataframe, add names (X, Y)
#     df=pd.DataFrame(data, columns=['X', 'Y'])
#     cause='X'
#     effect='Y'
# elif args.dataType.lower()=='lorenz' or args.dataType.lower()=='l':
#     data=gen_Lorenz(noiseType=args.noiseType, noiseWhen=args.noiseWhen, noiseAddType=args.noiseAddType, noiseLevel=args.noiseLevel, L=totalL)
#     # wrap as pandas dataframe, add names (X, Y, Z)
#     df=pd.DataFrame(data, columns=['X', 'Y', 'Z']) # for the bivar CCM we only compare X and Z (uni-directional) 
#     cause='X'
#     effect='Z'
# elif args.dataType.lower()=='rosslerlorenz' or args.dataType.lower()=='rl':
#     data=gen_RosslerLorenz(noiseType=args.noiseType, noiseWhen=args.noiseWhen, noiseAddType=args.noiseAddType, noiseLevel=args.noiseLevel, L=totalL)
#     # wrap as pandas dataframe, add names (X1, Y1, Z1, X2, Y2, Z2)
#     df=pd.DataFrame(data, columns=['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']) # for the bivarCCM we only compare X1 and X2 (uni-directional)
#     cause='X1'
#     effect='X2'

# read data, since it is already generated
data_dir=os.path.join(root, 'data', args.dataType)
if args.noiseType==None or args.noiseType.lower()=='none':
    file_name=os.path.join(data_dir, 'noNoise.csv')
else:
    file_name=os.path.join(data_dir, args.noiseType+"_"+args.noiseWhen+"_"+args.noiseAddType+"_"+str(args.noiseLevel)+'.csv')
    
df=pd.read_csv(file_name)

if args.dataType.lower()=='bivarlogistic' or args.dataType.lower()=='bilog':
    cause='X'
    effect='Y'
elif args.dataType.lower()=='lorenz' or args.dataType.lower()=='l':
    cause='X'
    effect='Z'
elif args.dataType.lower()=='rosslerlorenz' or args.dataType.lower()=='rl':
    cause='X1'
    effect='X2'

# downsample
if args.downsampleType!=None and args.downsampleType.lower()!='none':
    if args.downsampleType.lower() in ['a', 'av', 'average']:
        df=df.groupby(np.arange(len(df))//args.downsampleFactor).mean()
    elif args.downsampleType.lower() in ['d', 'de', 'decimation']:
        df=df.iloc[::args.downsampleFactor]
    elif args.downsampleType.lower() in ['s', 'sub', 'subsample']:
        df=df.groupby(np.arange(len(df))//args.downsampleFactor).apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)
    else:
        raise ValueError('Unknown downsampleType')

totalL=len(df)

# save path - add subfolder for the data type
output_dir=os.path.join(output_dir, args.dataType)
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
        
# save path - add subfolder for tau and emd
output_dir=os.path.join(output_dir, 'tau'+str(args.tau)+'_emd'+str(args.emd))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save path - add subfolder for the noise type
if args.noiseType==None or args.noiseType.lower()=='none':
    output_dir=os.path.join(output_dir, 'noNoise')
else:
    output_dir=os.path.join(output_dir, args.noiseType, args.noiseWhen+"_"+args.noiseAddType+"_"+str(args.noiseLevel))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# # range of L
# stepL=int(args.maxL/args.numL)
# range_L=np.arange(stepL, args.maxL+1, stepL)
# arr_sc1=np.zeros((args.numSeeds, args.numL))
# arr_sc2=np.zeros((args.numSeeds, args.numL))

# maxL and stepL depending on whether there is downsampling
if args.downsampleType is None or args.downsampleType.lower() == 'none':
    maxL = args.maxL
    numL = args.numL
    stepL = int(maxL/numL)
    range_L = np.arange(stepL, maxL+1, stepL)
else:
    maxL = args.maxL//args.downsampleFactor
    numL = args.numL
    stepL = int(maxL/numL)
    range_L = np.arange(stepL, maxL+1, stepL)

arr_sc1=np.zeros((args.numSeeds, numL))
arr_sc2=np.zeros((args.numSeeds, numL))


for seed in range(0, args.numSeeds):
    random.seed(seed*3)
    np.random.seed(seed*3)
    for idxL in range(len(range_L)):
        L=range_L[idxL]
        # generate a start point
        start_point=np.random.randint(0, totalL-L)
        # load data
        df_crop=df.iloc[start_point:start_point+L+1]
        bivar_CCM=CM_simplex(df_crop, [cause], [effect], tau=args.tau, emd=args.emd, L=L)
        sc1_error, sc2_error, sc1_corr, sc2_corr=bivar_CCM.causality()
        arr_sc1[seed, idxL]=sc1_corr
        arr_sc2[seed, idxL]=sc2_corr
        # # temporarily save the results
        # np.save(os.path.join(output_dir, 'arr_sc1.npy'), arr_sc1)
        # np.save(os.path.join(output_dir, 'arr_sc2.npy'), arr_sc2)
np.save(os.path.join(output_dir, 'arr_sc1.npy'), arr_sc1)
np.save(os.path.join(output_dir, 'arr_sc2.npy'), arr_sc2)

# visualize by plotting: average of sc over seeds - L, also plot the std as error bar
plt.figure()
plt.errorbar(range_L, np.mean(arr_sc1, axis=0), yerr=np.std(arr_sc1, axis=0), label='sc1: '+cause+'->'+effect)
plt.errorbar(range_L, np.mean(arr_sc2, axis=0), yerr=np.std(arr_sc2, axis=0), label='sc2: '+effect+'->'+cause)
plt.xlabel('L')
plt.ylabel('sc')
if args.noiseType==None or args.noiseType.lower()=='none':
    plt.title(args.dataType+' no noise')
else:
    plt.title(args.dataType+' '+args.noiseType+' '+args.noiseWhen+' '+args.noiseAddType+' '+str(args.noiseLevel))
plt.legend()
plt.savefig(os.path.join(output_dir, 'L_vs_avgCorr.png'))
plt.close()