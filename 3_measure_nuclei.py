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

args = parser.parse_args()
print(args.dataPath)

if args.dataPath is None:
    print("Please provide a data path")
    exit(1)

def get_corrected_shape_measurements(bbox_slice, image_nucleus_channel, image_nhsester_channel, nucleus_threshold):
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
    '''
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
        "nhsester_min_intensity": nhsester_min_intensity
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


def get_shannon_entropy_mask(image, label_mask, label_id):
    '''
    Calculate the shannon entropy of the image masked by the object (nuclei or other)
    return the shannon entropy of the masked image
    '''
    binary_mask = label_mask == label_id
    masked_image = image * binary_mask
    return shannon_entropy(masked_image)


def main(datapath='.', extension='.tif', compute_dask_data=True, resolution_level=None, min_voxel_volume=1000, sigma_gaussian=2):
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
    nuclei_channel_normalized = (nuclei_channel - nuclei_channel.min()) / (nuclei_channel.max() - nuclei_channel.min())
    nhsester_channel_normalized = (nhsester_channel - nhsester_channel.min()) / (nhsester_channel.max() - nhsester_channel.min())

    # Nuclei segmentation
    nuclei_gauss = gaussian_filter(nuclei_channel_normalized, sigma=sigma_gaussian)

    threshold_nuclei_otsu = threshold_otsu(nuclei_gauss)
    nuclei_mask = nuclei_gauss > threshold_nuclei_otsu
    nuclei_labels = label(nuclei_mask)

    print(f"Found {nuclei_labels.max()} objects in the nuclei channel, before filtering")    
    
    nuclei_labels_filtered = remove_small_objects(nuclei_labels, min_size=min_voxel_volume)
    print(f"Found {np.unique(nuclei_labels_filtered).size - 1} objects in the nuclei channel, after filtering with min size {min_voxel_volume} voxels")

    measurements = regionprops_table(
        nuclei_labels_filtered,
        intensity_image=nuclei_channel_normalized,
        properties=[
            'label',
            'area',
            'area_bbox',
            'area_convex',
            'bbox',
            'centroid',
            'intensity_mean',
            'intensity_max',
            'intensity_min',
            'intensity_std',
            'num_pixels',
            'slice',
            'axis_major_length',
            'axis_minor_length',
            'moments',
            'moments_central',
            'euler_number',
            'solidity'
        ],
        spacing=tuple(pixel_sizes)
    )
    measurements_df = pd.DataFrame(measurements)

    print("Calculating corrected shape measurements for each nucleus...")

    measurements_df[['area_corrected', 'nucleus_total_area', 'euler_number_corrected', 'euler_number_nucleoli_corrected', 'solidity_corrected', 'area_nucleolus_corrected', 'dna_volume_fraction', 'nucleolus_volume_fraction', 'distance_to_center', 'distance_to_border', 'shannon_entropy_nuclei', 'shannon_entropy_nhsester', 'nhsester_mean_intensity', 'nhsester_std_intensity', 'nhsester_max_intensity', 'nhsester_min_intensity']] = np.nan

    for row in tqdm(measurements_df.itertuples(), total=len(measurements_df), desc="Calculating corrected shape measurements"):
        bbox_slice = row.slice
        centroid = [measurements_df.at[row.Index, 'centroid-0'], measurements_df.at[row.Index, 'centroid-1'], measurements_df.at[row.Index, 'centroid-2']]
        corrected_stats = get_corrected_shape_measurements(bbox_slice, nuclei_channel_normalized, nhsester_channel_normalized, threshold_nuclei_otsu)
        measurements_df.at[row.Index, 'nucleus_total_area'] = corrected_stats['nucleus_total_area']
        measurements_df.at[row.Index, 'area_corrected'] = corrected_stats['area_corrected']
        measurements_df.at[row.Index, 'euler_number_corrected'] = corrected_stats['euler_number_corrected']
        measurements_df.at[row.Index, 'solidity_corrected'] = corrected_stats['solidity_corrected']
        measurements_df.at[row.Index, 'dna_volume_fraction'] = corrected_stats['dna_volume_fraction']
        measurements_df.at[row.Index, 'distance_to_center'] = distance_to_image_center(centroid, image_array.shape[1:], pixel_sizes)
        measurements_df.at[row.Index, 'distance_to_border'] = distance_to_image_border(centroid, image_array.shape[1:], pixel_sizes)
        measurements_df.at[row.Index, 'shannon_entropy_nuclei'] = shannon_entropy(nuclei_channel_normalized[bbox_slice])
        # measurements_df.at[row.Index, 'shannon_entropy_mask_nuclei'] = get_shannon_entropy_mask(nuclei_channel, nuclei_labels_filtered, row.label)
        measurements_df.at[row.Index, 'nhsester_mean_intensity'] = corrected_stats['nhsester_mean_intensity']
        measurements_df.at[row.Index, 'nhsester_std_intensity'] = corrected_stats['nhsester_std_intensity']
        measurements_df.at[row.Index, 'nhsester_max_intensity'] = corrected_stats['nhsester_max_intensity']
        measurements_df.at[row.Index, 'nhsester_min_intensity'] = corrected_stats['nhsester_min_intensity']
        measurements_df.at[row.Index, 'shannon_entropy_nhsester'] = shannon_entropy(nhsester_channel_normalized[bbox_slice])
        measurements_df.at[row.Index, 'euler_number_nucleoli_corrected'] = corrected_stats['euler_number_nucleoli_corrected']
        measurements_df.at[row.Index, 'area_nucleolus_corrected'] = corrected_stats['area_nucleolus_corrected']
        measurements_df.at[row.Index, 'nucleolus_volume_fraction'] = corrected_stats['nucleolus_volume_fraction']

    print(f"Saving measurements to {save_path}")
    measurements_df.to_csv(save_path / f"{filename}_nuclei_measurements_reslevel_{resolution_level}.csv")
    print("Done.")

if __name__ == "__main__":
    main(datapath=args.dataPath, extension=args.extension, compute_dask_data=args.computeDaskData, resolution_level=args.resolutionLevel, min_voxel_volume=args.minVoxelVolume, sigma_gaussian=args.sigmaGaussian)