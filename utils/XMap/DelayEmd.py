import numpy as np


def time_delay_embed(ts, tau, emd, L=None):
    """ Time-delay embedding of a univariate time series.
    Args:
        ts   time series (univariate)
        tau  time delay
        emd  embedding dimension
        L    max length/time length, if None then full length
        
    Returns:
        An numpy array. Shape: [L-(emd-1)*tau, emd].

        

        Each row is one sample that represents a point (a certain time index t) on the embeddng;
        the elements of each sample are the values at time indices: 
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold
        
        Array dimensions: [number of samples, embedding dimension].
        Number of samples equals L-(emd-1)*tau.
    """

    if L is None:
        L = len(ts)
    resultArr=np.zeros((L-(emd-1)*tau,emd))
    for t in range((emd - 1) * tau, L):
        for i in range(emd):
            resultArr[t - (emd - 1) * tau][i] = ts[t - i * tau]
    return resultArr

