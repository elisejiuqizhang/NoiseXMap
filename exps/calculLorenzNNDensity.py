import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

# Function to load data based on noiseType, noiseWhen, noiseAddType, and noiseLevel
def load_noise_data(noiseType, noiseWhen, noiseAddType, noiseLevel, data_dir):
    if noiseLevel == 0:
        file_name = 'noNoise.csv'
    else:
        file_name = f"{noiseType}_{noiseWhen}_{noiseAddType}_{round(noiseLevel, 2)}.csv"
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)

# Function to create time delay embeddings
def create_delay_embedding(series, delay=1, dimension=3):
    n = len(series)
    if n < (dimension - 1) * delay + 1:
        raise ValueError("Time series is too short for the given delay and dimension.")
    embedding = np.zeros((n - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        embedding[:, i] = series[i * delay: n - (dimension - 1 - i) * delay]
    return embedding

# Function to formulate the manifold embeddings for X, Y, and Z
def formulate_manifolds(data_frame, delay):
    X = data_frame['X'].values
    Y = data_frame['Y'].values
    Z = data_frame['Z'].values
    return {
        'X': create_delay_embedding(X, delay),
        'Y': create_delay_embedding(Y, delay),
        'Z': create_delay_embedding(Z, delay),
        'GroundTruth': np.vstack((X, Y, Z)).T
    }

# Function to calculate nearest neighbors
def calculate_nearest_neighbors(embedding, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    return distances, indices

# Function to retrieve time indices from neighbor indices
def retrieve_time_indices(neighbor_indices):
    return neighbor_indices.flatten()

# Function to find corresponding points on other manifolds
def find_corresponding_points(manifolds, time_indices):
    corresponding_points = {}
    for key, manifold in manifolds.items():
        # Ensure time_indices are within bounds
        valid_indices = time_indices[time_indices < len(manifold)]
        corresponding_points[key] = manifold[valid_indices]
    return corresponding_points

# Function to calculate density estimates using Euclidean distances in 3D
def calculate_density(points, n_neighbors, epsilon=1e-10):
    if points.shape[1] != 3:
        raise ValueError("Points must be in 3D space for Euclidean distance calculation.")
    # Ensure n_neighbors does not exceed the number of points
    n_neighbors = min(n_neighbors, len(points))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)
    distances = np.maximum(distances, epsilon)  # Avoid division by zero
    density = np.mean(1 / np.linalg.norm(distances, axis=1))
    return density

# Function to calculate densities for neighborhoods from a source manifold
def calculate_densities_from_source(manifolds, source_key, n_neighbors):
    source_embedding = manifolds[source_key]
    _, indices = calculate_nearest_neighbors(source_embedding, n_neighbors)
    densities = []
    for i, neighbors in enumerate(indices):
        time_indices = retrieve_time_indices(neighbors)
        corresponding_points = find_corresponding_points(manifolds, time_indices)
        density_row = {'Index': i}
        density_row[f'{source_key}_Density'] = calculate_density(source_embedding[neighbors], n_neighbors)
        for key, points in corresponding_points.items():
            if key != source_key and points.size > 0:
                density_row[f'{key}_Density'] = calculate_density(points, n_neighbors)
            else:
                density_row[f'{key}_Density'] = np.nan  # Use NaN for empty points
        densities.append(density_row)
    return densities

# Main function to execute the entire process and save results
def main():
    parser = argparse.ArgumentParser(description='Run Lorenz system density calculations.')
    parser.add_argument('--noiseType', type=str, required=True, choices=['gNoise', 'lpNoise'], help='Type of noise')
    parser.add_argument('--noiseWhen', type=str, required=True, choices=['in', 'post'], help='When noise is added')
    parser.add_argument('--noiseAddType', type=str, required=True, choices=['add', 'mult', 'both'], help='Type of noise addition')
    parser.add_argument('--noiseLevel', type=float, required=True, help='Noise level')
    parser.add_argument('--delay', type=int, required=True, help='Delay for time embeddings')
    parser.add_argument('--n_neighbors', type=int, required=True, help='Number of nearest neighbors')
    parser.add_argument('--data_dir', type=str, default='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/NoiseXMap/data/Lorenz', help='Directory with input data')
    parser.add_argument('--output_dir', type=str, default='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/NoiseXMap/outputs/LorenzNNDensity', help='Directory to save the output results')

    args = parser.parse_args()
    
    df = load_noise_data(args.noiseType, args.noiseWhen, args.noiseAddType, args.noiseLevel, args.data_dir)
    manifolds = formulate_manifolds(df, args.delay)
    
    output_subdir = os.path.join(args.output_dir, f"{args.noiseType}_{args.noiseWhen}_{args.noiseAddType}_{round(args.noiseLevel, 2)}_delay{args.delay}_nn{args.n_neighbors}")
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    
    output_files = ['NNFromGroundTruth-AllOtherThree.csv', 'NNFromDEX-AllOtherThree.csv', 'NNFromDEY-AllOtherThree.csv', 'NNFromDEZ-AllOtherThree.csv']
    source_keys = ['GroundTruth', 'X', 'Y', 'Z']
    
    for source_key, output_file in zip(source_keys, output_files):
        try:
            densities = calculate_densities_from_source(manifolds, source_key, args.n_neighbors)
        except IndexError as e:
            print(f"Skipping {source_key} due to IndexError: {e}")
            continue

        df_output = pd.DataFrame(densities)
        df_output.to_csv(os.path.join(output_subdir, output_file), index=False)

if __name__ == "__main__":
    main()
