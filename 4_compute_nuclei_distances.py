import os
from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataPath", help="The path to your data, should be tables with nuclei measurements")
parser.add_argument("--pixelInfoPath", help="The path to the pixel info csv file, should contain columns 'filename', 'pixel_size_x', 'pixel_size_y', 'pixel_size_z'", default=None)

args = parser.parse_args()
print(args.dataPath)

if args.dataPath is None:
    print("Please provide a data path")
    exit(1)


def euclidean_distance_scaled(df, cell1_idx, cell2_idx, pixel_scale):
    '''
    Calculate the euclidean distance between two cells, in microns.
    df: dataframe with the nuclei measurements, should contain columns 'centroid-0', 'centroid-1', 'centroid-2'
    cell1_idx: index of first cell to compare
    cell2_idx: index of second cell to compare
    pixel_scale: list or array with the pixel size in microns for each dimension, should be in the order [z, y, x]
    '''
    cell1 = [df.at[cell1_idx, "centroid-0"], df.at[cell1_idx, "centroid-1"], df.at[cell1_idx, "centroid-2"]]
    cell2 = [df.at[cell2_idx, "centroid-0"], df.at[cell2_idx, "centroid-1"], df.at[cell2_idx, "centroid-2"]]
    return ((pixel_scale[0] * (cell1[0] - cell2[0])) ** 2 + (pixel_scale[1] * (cell1[1] - cell2[1])) ** 2 + (pixel_scale[2] * (cell1[2] - cell2[2])) ** 2) ** 0.5


def euclidean_distance(df, cell1_idx, cell2_idx):
    '''
    Calculate the euclidean distance between two cells, in microns.
    df: dataframe with the nuclei measurements, should contain columns 'centroid-0', 'centroid-1', 'centroid-2'
    cell1_idx: index of first cell to compare
    cell2_idx: index of second cell to compare
    '''
    cell1 = [df.at[cell1_idx, "centroid-0"], df.at[cell1_idx, "centroid-1"], df.at[cell1_idx, "centroid-2"]]
    cell2 = [df.at[cell2_idx, "centroid-0"], df.at[cell2_idx, "centroid-1"], df.at[cell2_idx, "centroid-2"]]
    return ((cell1[0] - cell2[0]) ** 2 + (cell1[1] - cell2[1]) ** 2 + (cell1[2] - cell2[2]) ** 2) ** 0.5


def compute_closest_neighbors(distance_array):
    '''
    Computes the average distance to the closest neighbors for each cell  by doing gaussian mixture modeling and obtaining the leftmost component mean.
    distance_array: array of distances between cells
    '''
    distances_subset = np.array(distance_array[:30]) #TODO: best approach to choose number of elements

    data_reshaped = distances_subset.reshape(-1, 1) # reshape to 2D for sklearn
    best_n = None
    silhouette_scores = []
    for n in range(2, 10):
        gmm = GaussianMixture(n_components=n, random_state=42)
        labels = gmm.fit_predict(data_reshaped)
        score = silhouette_score(data_reshaped, labels)
        silhouette_scores.append(score)
    
    best_n = range(2, 10)[np.argmax(silhouette_scores)]

    gmm = GaussianMixture(n_components=best_n, random_state=42)
    gmm.fit(data_reshaped)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    # weights = gmm.weights_.flatten()

    return means.min(), stds[np.argmin(means)]


def main(datapath='.', pixelInfoPath=None):
    data_path = Path(datapath)
    filelist = os.listdir(data_path)
    filelist = [f for f in filelist if f.endswith('.csv')]
    
    pixel_information_path = Path(pixelInfoPath) if pixelInfoPath is not None else data_path.parent
    pixel_information_filelist = os.listdir(pixel_information_path)
    pixel_information_tablename = [f for f in pixel_information_filelist if 'pixel_sizes' in f and f.endswith('.csv')][0]
    print(f"Reading pixel information from: {pixel_information_tablename}")

    df_pixel_info = pd.read_csv(pixel_information_path / pixel_information_tablename)

    for filename in filelist:
        print(f"Processing {filename}...")

        original_filename = filename[:filename.index('_nuclei_measurements')]
        pixel_scale = df_pixel_info[df_pixel_info['filename'] == original_filename][['pixel_size_0', 'pixel_size_1', 'pixel_size_2']].values[0]
        print(f"Pixel scale for: {pixel_scale}")

        dataframe = pd.read_csv(data_path / filename)
        dataframe[[
            'area_scaled', 
            'area_corrected_scaled', 
            'distance_closest_neighbor', 
            'distance_avg_closest_neighbor', 
            'distance_std_closest_neighbor', 
            'aspect_ratio',
            'nucleus_total_area_scaled',
            'area_nucleolus_corrected_scaled',
        ]] = np.nan

        print("Calculating scaled volume (in microns^3) and distances...")
        pixel_volume = pixel_scale[0] * pixel_scale[1] * pixel_scale[2]
        for row in tqdm(dataframe.itertuples(), total=len(dataframe)):
            if row.area == row.num_pixels:
                volume = row.area * pixel_volume
                dataframe.at[row.Index, 'area_scaled'] = volume
            else:
                volume = row.area
            volume_corrected = row.area_corrected * pixel_volume
            dataframe.at[row.Index, 'area_corrected_scaled'] = volume_corrected

            dataframe.at[row.Index, 'aspect_ratio'] = row.axis_major_length / row.axis_minor_length if row.axis_minor_length > 0 else np.nan
            dataframe.at[row.Index, 'nucleus_total_area_scaled'] = row.nucleus_total_area * pixel_volume
            dataframe.at[row.Index, 'area_nucleolus_corrected_scaled'] = row.area_nucleolus_corrected * pixel_volume

            if volume_corrected > 500.: #TODO: better approach to set threshold, rejects small segmentation mistakes
                distances = []
                for other_row in dataframe.itertuples():
                    if row.Index != other_row.Index and other_row.area_corrected > 500.:
                        distance_scaled = euclidean_distance(dataframe, row.Index, other_row.Index)
                        distances.append(distance_scaled)
                
                distances.sort()
                dataframe.at[row.Index, 'distance_closest_neighbor'] = distances[0]
                avg_closest_neighbors, std_closest_neighbors = compute_closest_neighbors(distances)
                dataframe.at[row.Index, 'distance_avg_closest_neighbor'] = avg_closest_neighbors
                dataframe.at[row.Index, 'distance_std_closest_neighbor'] = std_closest_neighbors
            else:
                dataframe.at[row.Index, 'distance_closest_neighbor'] = np.nan
                dataframe.at[row.Index, 'distance_avg_closest_neighbor'] = np.nan
                dataframe.at[row.Index, 'distance_std_closest_neighbor'] = np.nan
        
        print(f"Saving measurements to {data_path / filename}")
        dataframe.to_csv(data_path / f"{filename}", index=False)
        print("=============")
    
    print("Done.")

if __name__ == "__main__":
    main(datapath=args.dataPath, pixelInfoPath=args.pixelInfoPath)