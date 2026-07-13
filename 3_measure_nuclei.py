import os
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.measure._regionprops")

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, regionprops_table, label, shannon_entropy
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

def get_corrected_shape_measurements(bbox_slice, image_nucleus_channel, image_nhsester_channel, nucleus_threshold, image_props=None):
    '''
    Get corrected shape measurements for a nucleus by combining the information from the nucleus channel and the nhsester channel. The corrected shape measurements are calculated by creating a combined (union) mask of the nucleus and nhsester channels. The corrected shape measurements include solidity, euler number, area, area of nhsester channel, dna area fraction and nhsester area fraction.
    returns: a dictionary with the following keys and values:
    - area_corrected: volume of DNA in unblurred data
    - nucleus_total_area: volume of the combined mask convex hull
    - euler_number_corrected: euler number of DNA in unblurred data
    - euler_number_nucleoli_corrected: euler number of nhsester in unblurred data, which is the nucleolus
    - solidity_corrected: solidity of DNA area
    - area_nucleolus_corrected: volume of nhsester nucleoli
    - dna_volume_fraction: fraction of DNA volume in the combined mask convex hull
    - nucleolus_volume_fraction: fraction of nhsester volume in the combined mask convex hull
    - nhsester_mean_intensity: mean intensity of nhsester in the nucleus area
    - nhsester_std_intensity: std intensity of nhsester in the nucleus area
    - nhsester_max_intensity: max intensity of nhsester in the nucleus area
    - nhsester_min_intensity: min intensity of nhsester in the nucleus area
    - shannon_entropy_dna: shannon entropy of DNA in the nucleus area
    - shannon_entropy_nhsester: shannon entropy of nhsester in the nucleus
    '''
    if image_nhsester_channel is None and image_props is not None:
        nucleus_crop = image_nucleus_channel[image_props['resolution_level_higher']].squeeze()
        nucleus_crop = nucleus_crop[2][bbox_slice].compute()
        nhsester_crop = image_nucleus_channel[image_props['resolution_level_higher']].squeeze()
        nhsester_crop = nhsester_crop[0][bbox_slice].compute()
        nucleus_crop = (nucleus_crop - image_props['nucleus_channel_min']) / (image_props['nucleus_channel_max'] - image_props['nucleus_channel_min'])
        nhsester_crop = (nhsester_crop - image_props['nhsester_channel_min']) / (image_props['nhsester_channel_max'] - image_props['nhsester_channel_min'])
    else:
        nucleus_crop = image_nucleus_channel[bbox_slice]
        nhsester_crop = image_nhsester_channel[bbox_slice]

    nucleus_crop_mask = nucleus_crop > nucleus_threshold
    nucleus_crop_labels = label(nucleus_crop_mask)
    nucleus_crop_measurements = regionprops(nucleus_crop_labels, intensity_image=nucleus_crop)

    threshold_nhsester_otsu = threshold_otsu(nhsester_crop)
    nhsester_crop_mask = nhsester_crop > threshold_nhsester_otsu
    nhsester_crop_labels = label(nhsester_crop_mask)
    nhsester_crop_measurements = regionprops(nhsester_crop_labels, intensity_image=nhsester_crop)

    combined_mask = np.logical_or(nucleus_crop_mask, nhsester_crop_mask)
    combined_labels = label(combined_mask)
    combined_measurements = regionprops(combined_labels)

    nucleus_total_area = 0
    if len(combined_measurements) > 1:
        for measurement in combined_measurements:
            nucleus_total_area += measurement.area_convex
    else:
        nucleus_total_area = combined_measurements[0].area_convex

    # if multiple labels are found in a nucleus
    area_corrected = 0
    euler_number_corrected = 0
    if len(nucleus_crop_measurements) > 1:
        for measurement in nucleus_crop_measurements:
            area_corrected += measurement.area
            euler_number_corrected += measurement.euler_number
    else:
        area_corrected = nucleus_crop_measurements[0].area
        euler_number_corrected = nucleus_crop_measurements[0].euler_number
    
    solidity_corrected = area_corrected / nucleus_total_area if nucleus_total_area > 0 else 0
    dna_volume_fraction = area_corrected / nucleus_total_area if nucleus_total_area > 0 else 0

    area_nucleolus_corrected = 0
    euler_number_nucleoli_corrected = 0
    if len(nhsester_crop_measurements) > 1:
        for measurement in nhsester_crop_measurements:
            area_nucleolus_corrected += measurement.area
            euler_number_nucleoli_corrected += measurement.euler_number
    else:
        area_nucleolus_corrected = nhsester_crop_measurements[0].area
        euler_number_nucleoli_corrected = nhsester_crop_measurements[0].euler_number

    nhsester_mean_intensity = np.mean(nhsester_crop)
    nhsester_std_intensity = np.std(nhsester_crop)
    nhsester_max_intensity = np.max(nhsester_crop)
    nhsester_min_intensity = np.min(nhsester_crop)

    shannon_entropy_nuclei = shannon_entropy(nucleus_crop)
    shannon_entropy_nhsester = shannon_entropy(nhsester_crop)
    
    nucleolus_volume_fraction = area_nucleolus_corrected / nucleus_total_area if nucleus_total_area > 0 else 0
    return {
        "area_corrected": area_corrected,
        "nucleus_total_area": nucleus_total_area,
        "euler_number_corrected": euler_number_corrected,
        "solidity_corrected": solidity_corrected,
        "area_nucleolus_corrected": area_nucleolus_corrected,
        "euler_number_nucleoli_corrected": euler_number_nucleoli_corrected,
        "dna_volume_fraction": dna_volume_fraction,
        "nucleolus_volume_fraction": nucleolus_volume_fraction,
        "nhsester_mean_intensity": nhsester_mean_intensity,
        "nhsester_std_intensity": nhsester_std_intensity,
        "nhsester_max_intensity": nhsester_max_intensity,
        "nhsester_min_intensity": nhsester_min_intensity,
        "shannon_entropy_nuclei": shannon_entropy_nuclei,
        "shannon_entropy_nhsester": shannon_entropy_nhsester
    }


def distance_to_image_center(centroid, image_shape, pixel_scale):
    '''
    Calculate the distance of an object centroid to the center of the image. The distance is calculated in 3D and takes into account the pixel scale of the image.
    '''
    image_center = np.array(image_shape) / 2
    distance = np.linalg.norm((np.array(centroid) - image_center) * np.array(pixel_scale))
    return distance


def distance_to_image_border(centroid, image_shape, pixel_scale):
    '''
    Calculate the distance of an object centroid to the closest border of the image. The distance is calculated in 3D and takes into account the pixel scale of the image.
    '''
    distances_to_borders = [
        centroid[0] * pixel_scale[0], # distance to top border
        (image_shape[0] - centroid[0]) * pixel_scale[0], # distance to bottom border
        centroid[1] * pixel_scale[1], # distance to left border
        (image_shape[1] - centroid[1]) * pixel_scale[1], # distance to right border
        centroid[2] * pixel_scale[2], # distance to front border
        (image_shape[2] - centroid[2]) * pixel_scale[2] # distance to back border
    ]
    return min(distances_to_borders)


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
            properties=feature_properties
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


def main(datapath='.', extension='.tif', compute_dask_data=True, resolution_level=None, min_voxel_volume=1000, sigma_gaussian=2, compute_high_resolution_features=False):
    data_path = Path(datapath)
    filename = data_path.stem
    print(f"Processing {filename}...")
    # database_name = "name" # TODO: extract path and name from database

    current_dir = Path.cwd()
    save_path = current_dir / "nuclei_measurements"
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
        "resolution_level_higher": None,
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
        feature_properties = ['label', 'area', 'area_bbox', 'area_convex', 'bbox', 'centroid', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std', 'num_pixels', 'slice', 'axis_major_length', 'axis_minor_length', 'moments', 'moments_central', 'euler_number', 'solidity']

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

        #compute new properties for higher resolution
        image_processing_props['resolution_level_higher'] = resolution_level - 1 if resolution_level > 0 else 0
        resolution_difference_factor = 2 ** abs(image_processing_props['resolution_level'] - image_processing_props['resolution_level_higher'])
        image_processing_props['min_voxel_volume'] = min_voxel_volume * resolution_difference_factor  # Adjust min_voxel_volume for higher resolution
        image_processing_props['sigma_gaussian'] = sigma_gaussian * resolution_difference_factor  # Adjust sigma for higher resolution
        image_processing_props['pixel_sizes'] = [ps / resolution_difference_factor for ps in pixel_sizes]  # Adjust pixel sizes for higher resolution
        print(f"Adjusted pixel scale for higher resolution: {image_processing_props['pixel_sizes']}")
        image_processing_props['image_dims'] = dask_data[image_processing_props['resolution_level_higher']].shape[2:]  # Update image dimensions for higher resolution, exclude T and C dimensions # TODO: hard coded, should be obtained from metadata
        feature_properties = ['label', 'area', 'area_bbox', 'area_convex', 'bbox', 'centroid', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std', 'num_pixels', 'slice', 'axis_major_length', 'axis_minor_length', 'moments', 'moments_central', 'euler_number', 'solidity']

        measurements_df = high_resolution_nuclei_features(dask_data, nuclei_props, processing_props=image_processing_props, feature_properties=feature_properties)
    else:
        measurements_df = nuclei_props
        nuclei_props = None # clear var from memory

    print("Calculating corrected shape measurements for each nucleus...")

    measurements_df[['area_corrected', 'nucleus_total_area', 'euler_number_corrected', 'euler_number_nucleoli_corrected', 'solidity_corrected', 'area_nucleolus_corrected', 'dna_volume_fraction', 'nucleolus_volume_fraction', 'distance_to_center', 'distance_to_border', 'shannon_entropy_nuclei', 'shannon_entropy_nhsester', 'nhsester_mean_intensity', 'nhsester_std_intensity', 'nhsester_max_intensity', 'nhsester_min_intensity']] = np.nan

    for row in tqdm(measurements_df.itertuples(), total=len(measurements_df), desc="Calculating corrected shape measurements"):
        bbox_slice = row.slice
        centroid = [measurements_df.at[row.Index, 'centroid-0'], measurements_df.at[row.Index, 'centroid-1'], measurements_df.at[row.Index, 'centroid-2']]
        if compute_high_resolution_features == False:
            corrected_stats = get_corrected_shape_measurements(bbox_slice=bbox_slice, image_nucleus_channel=nuclei_channel_normalized, image_nhsester_channel=nhsester_channel_normalized, nucleus_threshold=threshold_nuclei_otsu, image_props=None)
        else:
            corrected_stats = get_corrected_shape_measurements(bbox_slice=bbox_slice, image_nucleus_channel=dask_data, image_nhsester_channel=None, nucleus_threshold=threshold_nuclei_otsu, image_props=image_processing_props)
        measurements_df.at[row.Index, 'nucleus_total_area'] = corrected_stats['nucleus_total_area']
        measurements_df.at[row.Index, 'area_corrected'] = corrected_stats['area_corrected']
        measurements_df.at[row.Index, 'euler_number_corrected'] = corrected_stats['euler_number_corrected']
        measurements_df.at[row.Index, 'solidity_corrected'] = corrected_stats['solidity_corrected']
        measurements_df.at[row.Index, 'dna_volume_fraction'] = corrected_stats['dna_volume_fraction']
        measurements_df.at[row.Index, 'distance_to_center'] = distance_to_image_center(centroid, image_processing_props['image_dims'], image_processing_props['pixel_sizes'])
        measurements_df.at[row.Index, 'distance_to_border'] = distance_to_image_border(centroid, image_processing_props['image_dims'], image_processing_props['pixel_sizes'])
        measurements_df.at[row.Index, 'shannon_entropy_nuclei'] = corrected_stats['shannon_entropy_nuclei']
        measurements_df.at[row.Index, 'nhsester_mean_intensity'] = corrected_stats['nhsester_mean_intensity']
        measurements_df.at[row.Index, 'nhsester_std_intensity'] = corrected_stats['nhsester_std_intensity']
        measurements_df.at[row.Index, 'nhsester_max_intensity'] = corrected_stats['nhsester_max_intensity']
        measurements_df.at[row.Index, 'nhsester_min_intensity'] = corrected_stats['nhsester_min_intensity']
        measurements_df.at[row.Index, 'shannon_entropy_nhsester'] = corrected_stats['shannon_entropy_nhsester']
        measurements_df.at[row.Index, 'euler_number_nucleoli_corrected'] = corrected_stats['euler_number_nucleoli_corrected']
        measurements_df.at[row.Index, 'area_nucleolus_corrected'] = corrected_stats['area_nucleolus_corrected']
        measurements_df.at[row.Index, 'nucleolus_volume_fraction'] = corrected_stats['nucleolus_volume_fraction']

    print(f"Saving measurements to {save_path}")
    if compute_high_resolution_features:
        measurements_df.to_csv(save_path / f"{filename}_nuclei_measurements_reslevel_{resolution_level}_highres_{image_processing_props['resolution_level_higher']}.csv")
    else:
        measurements_df.to_csv(save_path / f"{filename}_nuclei_measurements_reslevel_{resolution_level}.csv")
    print("Done.")

if __name__ == "__main__":
    main(datapath=args.dataPath, extension=args.extension, compute_dask_data=args.computeDaskData, resolution_level=args.resolutionLevel, min_voxel_volume=args.minVoxelVolume, sigma_gaussian=args.sigmaGaussian, compute_high_resolution_features=args.computeHighResolutionFeatures)