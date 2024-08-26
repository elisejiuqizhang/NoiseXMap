import os, sys
root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from utils.data_gen.BiVarLogistic import gen_BiVarLogistic_singleVarNoise
from utils.data_gen.Lorenz import gen_Lorenz_singleVarNoise
# from utils.data_gen.RosslerLorenz import gen_RosslerLorenz
import argparse

parser=argparse.ArgumentParser('generate data with single variable noise')

parser.add_argument('--dataType', type=str, default='BiVarLogistic', help='data type to use, options: "BiVarLogistic" ("BiLog"), "Lorenz" ("L"), "RosslerLorenz" ("RL")')
parser.add_argument('--noiseVar', type=str, default='X', help='variable to add noise to, options: "X", "Y", "Z"')
parser.add_argument('--noiseType', type=str, default='gNoise', help='noise type to use, options: None, "laplacian"/"lpNoise"/"l", "gaussian"/"gNoise"/"g"')
parser.add_argument('--noiseWhen', type=str, default='in', help='when to add noise, options: "in-generation"/"in", "post-generation"/"post", only effective when noiseType is not None')
parser.add_argument('--noiseAddType', type=str, default='add', help='additive or multiplicative noise, options: "additive"/"add", "multiplicative"/"mult", "both", only effective when noiseType is not None')
parser.add_argument('--noiseLevel', type=float, default=5e-2, help='noise level, only effective when noiseType is not None')

parser.add_argument('--L', type=int, default=10000, help="generation length")
args=parser.parse_args()

# save path 
output_dir=os.path.join(root, 'data', 'singleVarNoise', args.dataType)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save path - var being injected with noise
output_dir=os.path.join(output_dir, args.noiseVar)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save path - noise type
file_name=os.path.join(output_dir, args.noiseType+"_"+args.noiseWhen+"_"+args.noiseAddType+"_"+str(args.noiseLevel))

# generate data
if args.dataType.lower()=='bivarlogistic' or args.dataType.lower()=='bilog':
    data=gen_BiVarLogistic_singleVarNoise(a_x=3.7, a_y=3.72, b_xy=0.45, b_yx=0.25, noiseVar=args.noiseVar, noiseType=args.noiseType, noiseWhen=args.noiseWhen, noiseAddType=args.noiseAddType, noiseLevel=args.noiseLevel, L=args.L)
elif args.dataType.lower()=='lorenz' or args.dataType.lower()=='l':
    data=gen_Lorenz_singleVarNoise(noiseVar=args.noiseVar, noiseType=args.noiseType, noiseWhen=args.noiseWhen, noiseAddType=args.noiseAddType, noiseLevel=args.noiseLevel, L=args.L)

# first save as numpy array
np.save(file_name, data)

# then save as pandas dataframe
if args.dataType.lower()=='bivarlogistic' or args.dataType.lower()=='bilog':
    df=pd.DataFrame(data, columns=['X', 'Y'])
    df.to_csv(file_name+'.csv', index=True)
elif args.dataType.lower()=='lorenz' or args.dataType.lower()=='l':
    df=pd.DataFrame(data, columns=['X', 'Y', 'Z'])
    df.to_csv(file_name+'.csv', index=True)

    
