import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to load and process the density output files
def load_and_process_density_files(output_dir, noiseType, noiseWhen, noiseAddType, delay, n_neighbors, noiseLevels):
    results = []

    # Iterate over each subdirectory in the output directory
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            params = subdir.split('_')
            if (params[0] == noiseType and params[1] == noiseWhen and 
                params[2] == noiseAddType and int(params[4].replace('delay', '')) == delay and 
                int(params[5].replace('nn', '')) == n_neighbors):
                noise_level = float(params[3])
                if noise_level in noiseLevels:
                    density_data = {
                        'noise_level': noise_level
                    }
                    
                    for file in os.listdir(subdir_path):
                        if file.endswith('.csv'):
                            file_path = os.path.join(subdir_path, file)
                            df = pd.read_csv(file_path)
                            avg_densities = df.mean().to_dict()  # Compute average densities
                            density_data[file.split('.')[0]] = avg_densities
                    
                    results.append(density_data)
    
    return results

# Function to organize data by noise level and generate plots
def generate_plots(results, save_dir):
    # Organize data by noise level
    data_by_noise_level = {}
    for result in results:
        noise_level = result.pop('noise_level')
        for key, value in result.items():
            if key not in data_by_noise_level:
                data_by_noise_level[key] = []
            data_entry = {
                'noise_level': noise_level,
                'value': value
            }
            data_by_noise_level[key].append(data_entry)
    
    # Generate and save plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, data in data_by_noise_level.items():
        df = pd.DataFrame(data)
        df = df.sort_values('noise_level')
        plt.figure()
        for subkey in ['GroundTruth_Density', 'X_Density', 'Y_Density', 'Z_Density']:  # Include only relevant keys
            y_values = df['value'].apply(lambda x: x.get(subkey, np.nan)).dropna()
            plt.plot(df['noise_level'][:len(y_values)], y_values, marker='o', label=subkey)
        plt.title(key)
        plt.xlabel('Noise Level')
        plt.ylabel('Average Density')
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(save_dir, f'{key}.png')
        plt.savefig(plot_path)
        plt.close()

# Main function
def main():
    parser = argparse.ArgumentParser(description='Process density outputs and generate plots.')
    parser.add_argument('--output_dir', type=str, default='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/NoiseXMap/outputs/LorenzNNDensity', help='Directory with output results')
    parser.add_argument('--noiseType', type=str, default='gNoise', choices=['gNoise', 'lpNoise'], help='Type of noise')
    parser.add_argument('--noiseWhen', type=str, default='in', choices=['in', 'post'], help='When noise is added')
    parser.add_argument('--noiseAddType', type=str, default='add', choices=['add', 'mult', 'both'], help='Type of noise addition')
    parser.add_argument('--delay', type=int, default=1, help='Delay for time embeddings')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of nearest neighbors')
    parser.add_argument('--save_dir', type=str, default='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/NoiseXMap/outputs/LorenzNNDensity/viz', help='Directory to save the plots')

    args = parser.parse_args()

    noiseLevels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    results = load_and_process_density_files(
        args.output_dir, 
        args.noiseType, 
        args.noiseWhen, 
        args.noiseAddType, 
        args.delay, 
        args.n_neighbors,
        noiseLevels
    )
    generate_plots(results, args.save_dir)

if __name__ == "__main__":
    main()
