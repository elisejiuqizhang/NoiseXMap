# compare the outputs of mine and skccm on bivar logistic maps

import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import skccm
from skccm import CCM
from skccm.utilities import train_test_split

from utils.data_gen.BiVarLogistic import gen_BiVarLogistic
from utils.XMap.CM_simplex import CM_simplex

seqL=20000

list_seeds=[97, 197, 297, 397, 497, 597, 697, 797, 897, 997]

# generate data
data=gen_BiVarLogistic(a_x=3.7, a_y=3.72, b_xy=0.35, b_yx=0.1, noiseType=None, L=seqL)
dataX=data[:, 0]    
dataY=data[:, 1]

# wrap as pandas dataframe, add names (X, Y)
df=pd.DataFrame(data, columns=['X', 'Y'])



# my method
range_L=np.arange(500, 5001, 500)
arr_sc1_mine=np.zeros((len(list_seeds), len(range_L)))
arr_sc2_mine=np.zeros((len(list_seeds), len(range_L)))
for seed in list_seeds:    
    random.seed(seed)
    np.random.seed(seed)
    for idxL in range(len(range_L)):
        L=range_L[idxL]
        # generate a start point
        start_point=np.random.randint(0, seqL-L-10)
        # load data
        df_crop=df.iloc[start_point:start_point+L+10]
        bivar_CCM=CM_simplex(df_crop, ['X'], ['Y'], tau=2, emd=3, L=L)
        sc1_error, sc2_error, sc1_corr, sc2_corr=bivar_CCM.causality()
        arr_sc1_mine[list_seeds.index(seed), idxL]=sc1_corr
        arr_sc2_mine[list_seeds.index(seed), idxL]=sc2_corr
np.save('arr_sc1_mine.npy', arr_sc1_mine)
np.save('arr_sc2_mine.npy', arr_sc2_mine)
# visualize by plotting: average of sc over seeds - L, also plot the std as error bar
plt.figure()
plt.errorbar(range_L, np.mean(arr_sc1_mine, axis=0), yerr=np.std(arr_sc1_mine, axis=0), label='sc1: X->Y')
plt.errorbar(range_L, np.mean(arr_sc2_mine, axis=0), yerr=np.std(arr_sc2_mine, axis=0), label='sc2: Y->X')
plt.xlabel('L')
plt.ylabel('sc')
plt.title('my method')
plt.legend()
plt.savefig('my_method.png')
plt.close()


# skccm
num_Ls=10
arr_sc1_skccm=np.zeros((len(list_seeds), num_Ls))
arr_sc2_skccm=np.zeros((len(list_seeds), num_Ls))
for seed in list_seeds:
    random.seed(seed)
    np.random.seed(seed)
    
    # embed the vectors
    e1=skccm.Embed(dataX)
    e2=skccm.Embed(dataY)
    Xemd=e1.embed_vectors_1d(lag=2, embed=3)
    Yemd=e2.embed_vectors_1d(lag=2, embed=3)

    # train test split
    Xtr, Xte, Ytr, Yte = train_test_split(Xemd,Yemd, percent=.7)
    skCCM=CCM()

    # L range
    len_tr=len(Xtr)
    lib_lens=np.arange(len_tr/num_Ls, len_tr+1, len_tr/num_Ls, dtype='int')
    #test causation
    skCCM.fit(Xtr,Ytr)
    x1p, x2p = skCCM.predict(Xte, Yte,lib_lengths=lib_lens)
    sc1,sc2 = skCCM.score()   
    arr_sc1_skccm[list_seeds.index(seed), :]=sc1
    arr_sc2_skccm[list_seeds.index(seed), :]=sc2 
# save the results
np.save('arr_sc1_skccm.npy', arr_sc1_skccm)
np.save('arr_sc2_skccm.npy', arr_sc2_skccm)
# visualize by plotting: average of sc over seeds - L, also plot the std as error bar
plt.figure()
plt.errorbar(lib_lens, np.mean(arr_sc1_skccm, axis=0), yerr=np.std(arr_sc1_skccm, axis=0), label='sc1: X->Y')
plt.errorbar(lib_lens, np.mean(arr_sc2_skccm, axis=0), yerr=np.std(arr_sc2_skccm, axis=0), label='sc2: Y->X')
plt.xlabel('L')
plt.ylabel('sc')
plt.title('skccm')
plt.legend()
plt.savefig('skccm.png')
plt.close()
