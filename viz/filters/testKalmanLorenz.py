import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

output_dir=os.path.join(root, 'outputs', 'viz', 'filters', 'kalman')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# generate length
L=10000
# generation noise type
noiseType='gaussian'
noiseWhen='in'
noiseAddType='both'
noiseLevel=0.5
# the Kalman Filter
process_noise_std = 0.0001
measurement_noise_std = 0.5

# in-generation noise
def Lorenz_in(xyz, *, s=10, r=28, b=2.667, noiseType="None", noiseAddType=None, noiseLevel=None):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z

    if noiseType == None or noiseType.lower() == "none":
        return np.array([x_dot, y_dot, z_dot])
    else:
        lp_add = np.random.laplace(0, noiseLevel, xyz.shape)
        g_add = np.random.normal(0, noiseLevel, xyz.shape)
        lp_mult = np.random.laplace(1, noiseLevel, xyz.shape)
        g_mult = np.random.normal(1, noiseLevel, xyz.shape)
        if noiseType.lower() in ['laplacian', 'lap', 'l', 'lpNoise']:
            if noiseAddType.lower() in ["mult", "multiplicative"]:
                return np.array([x_dot, y_dot, z_dot]) * lp_mult
            elif noiseAddType.lower() in ['add', 'additive']:
                return np.array([x_dot, y_dot, z_dot]) + lp_add
            elif noiseAddType.lower() == "both":
                return np.array([x_dot, y_dot, z_dot]) * lp_mult + lp_add
        elif noiseType.lower() in ['gaussian', 'gaus', 'gNoise']:
            if noiseAddType.lower() in ["mult", "multiplicative"]:
                return np.array([x_dot, y_dot, z_dot]) * g_mult
            elif noiseAddType.lower() in ['add', 'additive']:
                return np.array([x_dot, y_dot, z_dot]) + g_add
            elif noiseAddType.lower() == "both":
                return np.array([x_dot, y_dot, z_dot]) * g_mult + g_add

# wrapper, generate data to a certain length L (default: 10000)
def gen_Lorenz(s=10, r=28, b=2.667, noiseType=None, noiseWhen='in', noiseAddType="add", noiseLevel=0.1, L=10000):
    data = np.zeros((L+1, 3))
    # initial conditions
    data[0] = np.random.rand(3)
    dt = 0.01
    if noiseType == None or noiseType.lower() == "none":
        for i in range(L):
            data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b) * dt
    else:  # with noise
        if noiseWhen.lower() in ["in", "in-generation"]:
            for i in range(L):
                data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel) * dt
        elif noiseWhen.lower() in ["post", "post-generation"]:
            for i in range(L):
                data[i+1] = data[i] + Lorenz_in(data[i], s=s, r=r, b=b) * dt
            lp_add = np.random.laplace(0, noiseLevel, data.shape)
            g_add = np.random.normal(0, noiseLevel, data.shape)
            lp_mult = np.random.laplace(1, noiseLevel, data.shape)
            g_mult = np.random.normal(1, noiseLevel, data.shape)
            if noiseType.lower() in ['laplacian', 'lap', 'l', 'lpNoise']:
                if noiseAddType.lower() in ["mult", "multiplicative"]:
                    return data * lp_mult
                elif noiseAddType.lower() in ['add', 'additive']:
                    return data + lp_add
                elif noiseAddType.lower() == "both":
                    return data * lp_mult + lp_add
            elif noiseType.lower() in ['gaussian', 'gaus', 'gNoise']:
                if noiseAddType.lower() in ["mult", "multiplicative"]:
                    return data * g_mult
                elif noiseAddType.lower() in ['add', 'additive']:
                    return data + g_add
                elif noiseAddType.lower() == "both":
                    return data * g_mult + g_add
    return data

# Wrapper function for Lorenz system for odeint
def lorenz_wrapper(xyz, t, s, r, b, noiseType=None, noiseAddType=None, noiseLevel=None):
    return Lorenz_in(xyz, s=s, r=r, b=b, noiseType=noiseType, noiseAddType=noiseAddType, noiseLevel=noiseLevel)

# Kalman Filter Implementation
def kalman_filter(measurements, dt, process_noise_std, measurement_noise_std,noiseType=None, noiseAddType=None, noiseLevel=None):
    # Initial state estimate
    x_hat = np.zeros((measurements.shape[0], 3))
    P = np.eye(3)  # Initial estimate covariance
    Q = process_noise_std**2 * np.eye(3)  # Process noise covariance
    R = measurement_noise_std**2 * np.eye(3)  # Measurement noise covariance
    H = np.eye(3)  # Measurement matrix
    I = np.eye(3)  # Identity matrix

    for k in range(1, measurements.shape[0]):
        # Time update (Prediction)
        x_hat_minus = odeint(lorenz_wrapper, x_hat[k-1], [0, dt], args=(10, 28, 2.667, noiseType, noiseAddType, noiseLevel))[-1]
        P_minus = P + Q

        # Measurement update (Correction)
        K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + R)  # Kalman gain
        x_hat[k] = x_hat_minus + K @ (measurements[k] - H @ x_hat_minus)
        P = (I - K @ H) @ P_minus

    return x_hat


# Generate Lorenz system data with noise
data = gen_Lorenz(noiseType=noiseType, noiseWhen=noiseWhen, noiseAddType=noiseAddType, noiseLevel=noiseLevel, L=L)
# Generate noise-free data also for plotting
data_noise_free = gen_Lorenz(noiseType=None, L=L)

# Apply the Kalman Filter
dt = 0.01
filtered_states = kalman_filter(data, dt, process_noise_std, measurement_noise_std)

# Plotting the results
plot_L = 3000
fig = plt.figure(figsize=(13, 13))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:plot_L, 0], data[:plot_L, 1], data[:plot_L, 2], c='r', marker='.', alpha=1, s=4, label='Noisy measurements')
ax.plot(data_noise_free[:plot_L, 0], data_noise_free[:plot_L, 1], data_noise_free[:plot_L, 2], 'g', linewidth=1, alpha=0.8, label='Noise-free data')
ax.plot(filtered_states[:plot_L, 0], filtered_states[:plot_L, 1], filtered_states[:plot_L, 2], 'b', linewidth=1, alpha=0.8, label='Extended Kalman Filter estimate')
ax.legend()
plt.show()
plt.savefig(os.path.join(output_dir,f'Lorenz_{noiseType}{noiseLevel}_{noiseWhen}_{noiseAddType}_p{process_noise_std}_m{measurement_noise_std}.png'))
