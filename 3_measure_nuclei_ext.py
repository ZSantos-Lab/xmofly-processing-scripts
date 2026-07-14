import os
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.measure._regionprops")

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, regionprops_table, label, shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import pandas as pd

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Node
from ome_zarr.utils import info

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--extension", help="The extension of the files to be processed", default='.zarr')
parser.add_argument("--computeDaskData", help="Load full data to memory or chunked with dask", default=True, type=bool)
parser.add_argument("--resolutionLevel", help="The resolution level to process, 0 for highest resolution. If left empty, prompt will ask for a new value.", default=None, type=int)
parser.add_argument("--minVoxelVolume", help="The minimum volume of objects to be considered in nb of voxels", default=1000, type=int)
parser.add_argument("--sigmaGaussian", help="The sigma for the gaussian filter applied to the nuclei channel before thresholding", default=2, type=float)
parser.add_argument("--computeHighResolutionFeatures", help="Compute high resolution nuclei features", default=False, type=bool)

args = parser.parse_args()
print(args.dataPath)

if args.dataPath is None:
    print("Please provide a data path")
    exit(1)


def get_high_res_slice(bbox_slice, resolution_level, resolution_level_higher):
    '''
    Compute corresponding slice in higher resolution from current resolution level and compute resolution level.
    params:
    - bbox_slice: the slice of the object in the current resolution level
    - resolution_level: the current resolution level
    - resolution_level_higher: the higher resolution level to compute the slice for
    returns: the slice of the object in the higher resolution level
    '''
    factor = 2 ** abs(resolution_level - resolution_level_higher)
    bbox_higher_res = tuple(slice(int(s.start * factor), int(s.stop * factor)) for s in bbox_slice)
    return bbox_higher_res


def high_resolution_nuclei_features(image_data_dask, feature_dataframe, processing_props=None, feature_properties=['label', 'area']):
    '''
    Get the coordinates (or slice) of nuclei in lower resolution image data and compute general shape parameters in higher resolution image data.
    params:
    - image_data_dask: the dask image data to process
    - feature_dataframe: dataframe with nuclei features and coordinates
    - processing_props: dictionary with processing properties (resolution_level, resolution_level_higher, sigma_gaussian, min_voxel_volume, nucleus_channel_min, nucleus_channel_max, nhsester_channel_min, nhsester_channel_max, pixel_sizes, threshold_nuclei_otsu)
    - feature_properties: list of properties to extract for each nucleus
    returns: a list of nuclei features in the high resolution image data from lower resolution coordinates.
    '''
    nhsester_channel = image_data_dask[processing_props['resolution_level_higher']][0][0] # channels are assumed to be in the order of [nhsester, other_channel, nuclei]
    nuclei_channel = image_data_dask[processing_props['resolution_level_higher']][0][2]

    nuclei_props_highres = []
    column_names = None

    for index, row in tqdm(feature_dataframe.iterrows(), total=feature_dataframe.shape[0], desc="Processing nuclei in high resolution"):
        original_label = row['label']
        bbox_slice = row['slice']
        bbox_slice_higher_res = get_high_res_slice(bbox_slice, processing_props['resolution_level'], processing_props['resolution_level_higher'])
        nucleus_crop = nuclei_channel[bbox_slice_higher_res].compute()
        nhsester_crop = nhsester_channel[bbox_slice_higher_res].compute()

        # Normalize the cropped images
        nucleus_crop_normalized = (nucleus_crop - processing_props['nucleus_channel_min']) / (processing_props['nucleus_channel_max'] - processing_props['nucleus_channel_min'])
        nhsester_crop_normalized = (nhsester_crop - processing_props['nhsester_channel_min']) / (processing_props['nhsester_channel_max'] - processing_props['nhsester_channel_min'])

        # Apply Gaussian filter to the nucleus channel
        nucleus_gauss = gaussian_filter(nucleus_crop_normalized, sigma=processing_props['sigma_gaussian'])

        # Thresholding
        if processing_props['threshold_nuclei_otsu'] is None:
            threshold_nuclei_otsu = threshold_otsu(nucleus_gauss)
        else:
            threshold_nuclei_otsu = processing_props['threshold_nuclei_otsu']

        nuclei_mask = nucleus_gauss > threshold_nuclei_otsu
        nuclei_labels = label(nuclei_mask)
        if np.unique(nuclei_labels).size - 1 > 1: # to avoid warnings when a single object is passed
            nuclei_labels_filtered = remove_small_objects(nuclei_labels, min_size=processing_props['min_voxel_volume'])
        else:
            nuclei_labels_filtered = nuclei_labels

        measurements = regionprops_table(
            nuclei_labels_filtered,
            intensity_image=nucleus_crop_normalized,
            properties=feature_properties,
            spacing=tuple(processing_props['pixel_sizes'])
        )
        # store as pandas dict to get the column names and the highest object in case of multiple objects in the cropped image
        nucleus_prop_df = pd.DataFrame(measurements)
        if column_names is None: column_names = nucleus_prop_df.columns.tolist()
        max_area_obj_index = nucleus_prop_df['area'].idxmax()
        nucleus_prop_df = nucleus_prop_df.iloc[max_area_obj_index]  # Keep only the largest object
        nucleus_prop_df['label'] = original_label
        nucleus_prop_df['slice'] = bbox_slice_higher_res  # Store the slice in the higher resolution image

        # Scale the centroid and bounding box coordinates to the higher resolution
        nucleus_prop_df['centroid-0'] = row['centroid-0'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['centroid-1'] = row['centroid-1'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['centroid-2'] = row['centroid-2'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['bbox-0'] = row['bbox-0'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['bbox-1'] = row['bbox-1'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['bbox-2'] = row['bbox-2'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['bbox-3'] = row['bbox-3'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['bbox-4'] = row['bbox-4'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))
        nucleus_prop_df['bbox-5'] = row['bbox-5'] * (2 ** abs(processing_props['resolution_level'] - processing_props['resolution_level_higher']))

        nuclei_props_highres.append(nucleus_prop_df.to_list())  # Convert the row to a list and append to the list of nuclei properties

    return pd.DataFrame(nuclei_props_highres, columns=column_names)


def get_gray_coocurrence_matrix(image_channel, bbox=None, distances=[0], angles=[0], levels=256):
    '''
    Calculate the gray level co-occurrence matrix (GLCM) for the given image channel within the bounding box slice.
    Due to size and resolution of analyzed datasets, distance values are set to be far from the computed pixel to capture patter differences.
    Same thing as with angles, to maximize the probability of capturing the pattern differences.
    returns the glcm
    '''
    glcm_properties = ['contrast', 'dissimilarity', 'homogeneity'] # more propertoes available: 'energy', 'correlation', 'ASM'
    if bbox is not None:
        image_crop = np.array(image_channel[bbox])
    else:
        image_crop = image_channel
    if len(image_crop.shape) > 2:
        image_crop = image_crop[image_crop.shape[0]//2, ...] # take the middle slice if 3D
        image_crop =  image_crop.squeeze()
    image_crop = (image_crop / image_crop.max() * (levels - 1)).astype('uint16') # rescale to the number of levels
    glcm = graycomatrix(image_crop, distances=distances, angles=angles, levels=levels)
    glcm_features = {prop: graycoprops(glcm, prop) for prop in glcm_properties}
    return {
        'contrast': glcm_features['contrast'],
        'dissimilarity': glcm_features['dissimilarity'],
        'homogeneity': glcm_features['homogeneity'],
    }


def main(datapath='.', extension='.tif', compute_dask_data=True, resolution_level=None, min_voxel_volume=1000, sigma_gaussian=2, compute_high_resolution_features=False):
    data_path = Path(datapath)
    filename = data_path.stem
    print(f"Processing {filename}...")
    # database_name = "name" # TODO: extract path and name from database

    current_dir = Path.cwd()
    save_path = current_dir / "nuclei_measurements_ext"
    if not save_path.exists():
        os.mkdir(save_path)
    

    if extension != '.zarr':
        print("Only .zarr files are supported for now.")
        exit(1)

    # Open the zarr file
    zarr_file = parse_url(datapath, mode='r')
    reader = Reader(zarr_file)
    nodes = list(reader())
    print(f"Found {len(nodes)} nodes in the zarr file")

    zarr_info = info(data_path)
    img_info = list(zarr_info)[0].data

    min_voxel_volume_input = None
    sigma_gaussian_input = None

    if resolution_level is None:
        # if resolution level is not set, ask user to input it along with minimum voxel volume and sigma for gaussian filter
        resolution_level = int(input("Please enter the resolution level to process (0 for highest resolution): "))
        min_voxel_volume_input = input("Please enter the minimum volume of objects to be considered in nb of voxels (e.g. 1000): ")
        sigma_gaussian_input = input("Please enter the sigma for the gaussian filter applied to the nuclei channel before thresholding (e.g. 2): ")
    
    if min_voxel_volume_input is not None and len(min_voxel_volume_input) > 0:
        min_voxel_volume = int(min_voxel_volume_input)
    if sigma_gaussian_input is not None and len(sigma_gaussian_input) > 0:
        sigma_gaussian = float(sigma_gaussian_input)
    
    print(f"Resolution level set: {resolution_level}; Minimum voxel volume: {min_voxel_volume}; Sigma for gaussian filter: {sigma_gaussian}")

    # Get the image data at the specified resolution level
    image_node = nodes[0]
    dask_data = image_node.data
    if compute_dask_data:
        image_array = dask_data[resolution_level].compute()
    else:
        image_array = dask_data[resolution_level]
    
    if len(image_array.shape) > 4:
        image_array = image_array.squeeze()
    print(f"Image shape: {image_array.shape}")

    # Read pixel scale from metadata
    metadata = image_node.metadata
    pixel_sizes = metadata['coordinateTransformations'][resolution_level][0]["scale"]
    pixel_sizes = pixel_sizes[-3:]
    print("Pixel scale (TCZYX)", pixel_sizes)

    nhsester_channel = image_array[0] # TODO: get channel index from metadata
    nuclei_channel = image_array[2]

    # image normalization
    nucleus_channel_min = nuclei_channel.min()
    nucleus_channel_max = nuclei_channel.max()
    nuclei_channel_normalized = (nuclei_channel - nucleus_channel_min) / (nucleus_channel_max - nucleus_channel_min)
    nhsester_channel_min = nhsester_channel.min()
    nhsester_channel_max = nhsester_channel.max()
    nhsester_channel_normalized = (nhsester_channel - nhsester_channel_min) / (nhsester_channel_max - nhsester_channel_min)


    # Nuclei segmentation
    nuclei_gauss = gaussian_filter(nuclei_channel_normalized, sigma=sigma_gaussian)

    threshold_nuclei_otsu = threshold_otsu(nuclei_gauss)
    nuclei_mask = nuclei_gauss > threshold_nuclei_otsu
    nuclei_labels = label(nuclei_mask)

    image_processing_props = {
        "resolution_level": resolution_level,
        "resolution_level_higher": 1, # for nuclei analysis purposes, highest resolution will be 1 for consistency
        "min_voxel_volume": min_voxel_volume,
        "sigma_gaussian": sigma_gaussian,
        "nucleus_channel_min": nucleus_channel_min,
        "nucleus_channel_max": nucleus_channel_max,
        "nhsester_channel_min": nhsester_channel_min,
        "nhsester_channel_max": nhsester_channel_max,
        "pixel_sizes": pixel_sizes,
        "threshold_nuclei_otsu": threshold_nuclei_otsu,
        "image_dims": image_array.shape[1:] # exclude channel dimension
    }

    print(f"Found {nuclei_labels.max()} objects in the nuclei channel, before filtering")    
    
    nuclei_labels_filtered = remove_small_objects(nuclei_labels, min_size=min_voxel_volume)
    print(f"Found {np.unique(nuclei_labels_filtered).size - 1} objects in the nuclei channel, after filtering with min size {min_voxel_volume} voxels")

    if compute_high_resolution_features:
        print(f"Computing high resolution features for each nucleus. Minimum voxel volume and sigma for gaussian filter will be adjusted for higher resolution.")
        feature_properties = ['label', 'area', 'bbox', 'centroid', 'slice']
    else:
        feature_properties = ['label', 'area', 'area_bbox', 'bbox', 'centroid', 'num_pixels', 'slice']

    measurements = regionprops_table(
        nuclei_labels_filtered,
        intensity_image=nuclei_channel_normalized,
        properties=feature_properties,
        spacing=tuple(pixel_sizes)
    )
    nuclei_props = pd.DataFrame(measurements)

    if compute_high_resolution_features:
        #clear unused vars from memory
        nuclei_channel_normalized = None
        nhsester_channel_normalized = None
        nuclei_labels_filtered = None
        image_array = None

        #compute new properties for higher resolution
        if image_processing_props['resolution_level_higher'] is None: image_processing_props['resolution_level_higher'] = resolution_level - 1 if resolution_level > 0 else 0
        resolution_difference_factor = 2 ** abs(image_processing_props['resolution_level'] - image_processing_props['resolution_level_higher'])
        image_processing_props['min_voxel_volume'] = min_voxel_volume * resolution_difference_factor  # Adjust min_voxel_volume for higher resolution
        image_processing_props['sigma_gaussian'] = sigma_gaussian * resolution_difference_factor  # Adjust sigma for higher resolution
        image_processing_props['pixel_sizes'] = [ps / resolution_difference_factor for ps in pixel_sizes]  # Adjust pixel sizes for higher resolution
        print(f"Adjusted pixel scale for higher resolution: {image_processing_props['pixel_sizes']}")
        image_processing_props['image_dims'] = dask_data[image_processing_props['resolution_level_higher']].shape[2:]  # Update image dimensions for higher resolution, exclude T and C dimensions # TODO: hard coded, should be obtained from metadata
        feature_properties = ['label', 'area', 'area_bbox', 'bbox', 'centroid', 'num_pixels', 'slice']

        measurements_df = high_resolution_nuclei_features(dask_data, nuclei_props, processing_props=image_processing_props, feature_properties=feature_properties)
    else:
        measurements_df = nuclei_props
        nuclei_props = None # clear var from memory

    print("Calculating extended shape measurements for each object...")

    measurements_df[['shannon_entropy_nuclei', 'shannon_entropy_nhsester']] = np.nan

    glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_distances = [3, 6, 12, 18]
    n_props = len(glcm_angles) * len(glcm_distances)

    for i in range(n_props):
        measurements_df[f'glcm_contrast_dna_{i}'] = np.nan
        measurements_df[f'glcm_dissimilarity_dna_{i}'] = np.nan
        measurements_df[f'glcm_homogeneity_dna_{i}'] = np.nan
        measurements_df[f'glcm_contrast_nhsester_{i}'] = np.nan
        measurements_df[f'glcm_dissimilarity_nhsester_{i}'] = np.nan
        measurements_df[f'glcm_homogeneity_nhsester_{i}'] = np.nan

    for row in tqdm(measurements_df.itertuples(), total=len(measurements_df), desc="Calculating extended shape measurements"):
        bbox_slice = row.slice
        if compute_high_resolution_features:
            image_data = dask_data[image_processing_props['resolution_level_higher']].squeeze()
            nucleus_crop_channel = image_data[2][bbox_slice].compute()
            nhsester_crop_channel = image_data[0][bbox_slice].compute()

            texture_stats_dna = get_gray_coocurrence_matrix(nucleus_crop_channel, bbox=None, distances=glcm_distances, angles=glcm_angles, levels=256)
            texture_stats_nhsester = get_gray_coocurrence_matrix(nhsester_crop_channel, bbox=None, distances=glcm_distances, angles=glcm_angles, levels=256)
            shannon_entropy_nuclei = shannon_entropy(nucleus_crop_channel)
            shannon_entropy_nhsester = shannon_entropy(nhsester_crop_channel)
        else:
            texture_stats_dna = get_gray_coocurrence_matrix(nuclei_channel, bbox=bbox_slice, distances=glcm_distances, angles=glcm_angles, levels=256)
            texture_stats_nhsester = get_gray_coocurrence_matrix(nhsester_channel, bbox=bbox_slice, distances=glcm_distances, angles=glcm_angles, levels=256)
            shannon_entropy_nuclei = shannon_entropy(nuclei_channel[bbox_slice])
            shannon_entropy_nhsester = shannon_entropy(nhsester_channel[bbox_slice])
        
        measurements_df.at[row.Index, 'shannon_entropy_nuclei'] = shannon_entropy_nuclei
        measurements_df.at[row.Index, 'shannon_entropy_nhsester'] = shannon_entropy_nhsester

        contrast_array = texture_stats_dna['contrast'].flatten()
        for i in range(len(contrast_array)):
            measurements_df.at[row.Index, f'glcm_contrast_dna_{i}'] = contrast_array[i]
        dissimilarity_array = texture_stats_dna['dissimilarity'].flatten()
        for i in range(len(dissimilarity_array)):
            measurements_df.at[row.Index, f'glcm_dissimilarity_dna_{i}'] = dissimilarity_array[i]
        homogeneity_array = texture_stats_dna['homogeneity'].flatten()
        for i in range(len(homogeneity_array)):
            measurements_df.at[row.Index, f'glcm_homogeneity_dna_{i}'] = homogeneity_array[i]
        
        contrast_array = texture_stats_nhsester['contrast'].flatten()
        for i in range(len(contrast_array)):
            measurements_df.at[row.Index, f'glcm_contrast_nhsester_{i}'] = contrast_array[i]
        dissimilarity_array = texture_stats_nhsester['dissimilarity'].flatten()
        for i in range(len(dissimilarity_array)):
            measurements_df.at[row.Index, f'glcm_dissimilarity_nhsester_{i}'] = dissimilarity_array[i]
        homogeneity_array = texture_stats_nhsester['homogeneity'].flatten()
        for i in range(len(homogeneity_array)):
            measurements_df.at[row.Index, f'glcm_homogeneity_nhsester_{i}'] = homogeneity_array[i]
        

    print(f"Saving measurements to {save_path}")
    if compute_high_resolution_features:
        measurements_df.to_csv(save_path / f"{filename}_nuclei_measurements_reslevel_{resolution_level}_highres_{image_processing_props['resolution_level_higher']}_extended.csv")
    else:
        measurements_df.to_csv(save_path / f"{filename}_nuclei_measurements_reslevel_{resolution_level}_extended.csv")
    print("Done.")

if __name__ == "__main__":
    main(datapath=args.dataPath, extension=args.extension, compute_dask_data=args.computeDaskData, resolution_level=args.resolutionLevel, min_voxel_volume=args.minVoxelVolume, sigma_gaussian=args.sigmaGaussian, compute_high_resolution_features=args.computeHighResolutionFeatures)