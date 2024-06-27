import numpy as np

def partial_corr(x, y, cond):
    """ Partial correlation between x and y conditionned on cond.
    ref: https://github.com/PengTao-HUST/crossmapy/blob/master/crossmapy/utils.py

    Parameters
    ----------
    x: 2D array [num_samples, dim]
        First variable.
    y: 2D array [num_samples, dim]
        Second variable.
    cond: 2D array [num_samples, dim]
        Conditioning variable.
    """  
    z = cond

    partial_corr = np.zeros(x.shape[1])

    for i in range(z.shape[1]):
        r_xy = np.corrcoef(x[:, i], y[:, i])[0, 1]
        r_xz = np.corrcoef(x[:, i], z[:, i])[0, 1]
        r_yz = np.corrcoef(y[:, i], z[:, i])[0, 1]

        partial_corr[i] = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

    return partial_corr


def corr(x, y):
    """ Pearson's correlation between x and y.
     2D array [num_samples, embed dim]
     compute correlation for each dimension then take average.
    """

    corr = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        corr[i] = np.corrcoef(x[:, i], y[:, i])[0, 1]

    return corr