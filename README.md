# airyscan-czi-subset
Repository with python scripts and notebooks to processes Airyscan data too large for processing in a limited RAM workstation. 

These scripts and pipelines are intended for pre-processing Airyscan raw data which is too large and cannot be processed as a single Z stack. They have been prepared for mosaic datasets, Zeiss Zen will process each tile independenlty.

The processing bottleneck is in the number of Z planes that can be processes. 
From my experience, it requires ~70 GB of RAM / 100 z planes of images with shape `(4098 x 4098)`. For a workstation with 256 GB of RAM, ~300 z planes can be processed at a time. In workstations with less RAM, one has to split Z stacks into smaller subsets.

The current workflow consists of splitting raw Airyscan data into substacks for Airyscan processing in Zeiss Zen, splitting processed data into the individual tiles, stitch then in 3D, then perform the final 2D/mosaic stitching for the complete dataset.
Stitching is done with the [`multiview stitcher`](https://github.com/multiview-stitcher/multiview-stitcher/tree/main) and using OME Zarr as the data format.

## Pipeline
1. Split RAW data into multiple sub-sets in `Zeiss Zen` with script
    - `Airyscan_subset_split_data.czmac` <sup>1</sup>
1. This will generate multiple raw substacks. Processed them individually or in Batch in `Zeiss Zen`
1. In `conda/mamba`, activate the corresponding environment and run scripts.
1. For Airyscan processed data, split processed files into individual Tiles
    - using: `splitsave_czi_tile.py`
1. Use the script to stitch individual subset Tiles in 3D to form full Tiles
    - using: `1_multiview_stitcher_3d.py`
1. Stitch full Tiles in 2D for the whole mosaic using files generated from previous script. __Note__: needs a `czi` file (raw or processed, does not matter) in it's parent folder to extract mosaic shape. The `czi` file needs metadata to extract this info. Usualy moving the `stitched_3d` files to the folder with `czi` files is enough.
    -using:" `2_multiview_stitcher_2d.py`

<!-- ![Schematic Pipeline](/media/schematic_processing_pipeline.png) -->
<img src="./media/schematic_processing_pipeline.png" alt="Schematic pipeline" width="300"/>

<sup>1</sup> - currently, this script only splits data into subsets of the z-stack, the script to split and process airyscan (`Airyscan_Subset_Split_process.czmac`) data has not been fully tested and may not generate the expected results.

## Python processing environment

### Create conda/mamba `env`
The python scripts require two packages: `pylibczirw` and `multiview-stitcher`, both available via Pypi. These packages have been tested with python version 3.10.

Installation: 
On your favored terminal application (`conda/mamba command prompt` in Windows or other terminal apps in other OSes)
```bash
# create a conda/mamba env
mamba create -n pyczi python=3.10

# activate the created env
mamba activate pyczi

# install needed dependencies
pip install pylibczirw multiview-stitcher
```

### Build from `yml` file
You can build the environment using our example file, however this environment is heavier and has many other packages used or tested during processing.

```bash
# create env using yml file
mamba env create -f /environments/pyczi_env_20250616_nobuilds.yml
```

## Usage
More detailed usage of each script and their arguments.

### Create Z-stack subset
script: `Airyscan_subset_split_data.czmac`

A special thanks to Anna Pezzarossa (Champalimaud Foundation) for providing the basis and the initial macro for processing this data.

__Motivation__: 
Airyscan processing in 3D requires plenty of RAM to process stacks. From my experience and from Anna Pezzarossa's feedback, a 256 GB RAM machine can process Z-stacks with around 300 z planes.
To process data with larger z-stacks, we need to split the datasets into subsets of the same data.

__Goal__: 
split data into smaller subsets to process Airyscan data and preserve metadata information

__How__:
We generated a Zeiss Zen Blue macro script to perform this action and split all datasets inside a folder. This script was generated with the help of a macro kindly provided by Anna Pezzarossa, which was adapted by me to do this task.

Name: [Airyscan_subset_split_data.czmac](https://github.com/nunopimpaomartins/airyscan-czi-subsample/blob/main/Airyscan_subset_split_data.czmac) 
How to use: 
1. open Zeiss Zen Blue in processing mode ("Zen Desk")
2. activate view of the Macro environment if not active
3. copy macro to Zeiss default folder: `D: drive\My Documents\Carl Zeiss\Zen\Documents\Macros\yourfolder\`
4. run macro 
5. **Important**: output data is not compressed by Default

Input:
```
input folder
├── file1.czi
├── file2.czi
├── file3.czi
└── file4.czi
```

Output:
```
input folder
├── split_czi
│   ├── file1-sub1-Scene1.czi
│   ├── file1-sub2-Scene1.czi
│   ├── file2-sub1-Scene1.czi
│   ├── file2-sub2-Scene1.czi
│   ├── file3-sub1-Scene1.czi
│   ├── file3-sub2-Scene1.czi
│   └── file3-sub3-Scene1.czi
├── file1.czi
├── file2.czi
├── file3.czi
└── file4.czi
```

___
### Process Airyscan data
Airyscan data processing needs to be done with Zeiss Zen with valid Airyscan processing license. There is a Fiji plugin which is able to do processing of airyscan data, but I have never tried it and it may require additional setup steps ([link](https://team.inria.fr/serpico/airyscanj/)).

These datasets can be processed either file by file (Single file mode) or in batch mode (Batch mode)

#### Single file mode
1. open Zeiss Zen Blue with valid license
2. in the left panel, choose `Single` mode
3. click and drag you file to the Zen window
4. below the image, choose the `Airycan processing` tab
5. choose the image mode: `3D processing` 
6. press create image.
7. once processing is finished, save image with compression in: `File > Save with options...`
8. Choose the `Lossless compression (ZSTD)` method and choose save name.

#### Batch mode
in batch mode it is possible to choose multiple files to process, which can be found in different folders and sub-folders. The output will be stored in the same folder as the input image.
1. open Zeiss Zen Blue with valid license
2. in the left panel, choose the `Batch` mode
3. still in the left panel, choose `Airyscan Processing` as the image processing step/algorithm
4. in the middle panel, click the `+` sign to add images and files to process
5. once the list is filled, press `Apply` in the left side panel to run the process
6. Once finished, open each processed file and resave with compression. See step 7 of [[#Single file mode]].

___
### Split Tiles
 script: `splitsave_czi_tile.py`

__Motivation__: 
the stitching tool Zeiss Zen Blue is not doing a good job at stitching the microscope data, so we need to test a different tool or approach. To do this, we need to split the tile data into different files to read in other tools.

__Goal__:
Split `czi` data into single `czi` tile files.

__How__:
Run the `python` file in the command prompt of `conda`/`mamba`  with an argument for the data path.
script name: [splitsave_czi_tile.py](https://github.com/nunopimpaomartins/airyscan-czi-subsample/blob/main/splitsave_czi_tile.py) 

How to use:
1. open a terminal or, if in windows, the `conda`/`mamba` 
2. browse to the script directory using `cd` 
```bash
$ cd path\to\script\
```
3. then run the script by adding the `--dataPath` argument followed by the folder path, and optionally, the file extensions with `--extension` 
```bash
# if using Mac or Linux
python splitsave_czi_tile.py --dataPath "/path/to/image/folder/"

# if using Windows
python splitsave_czi_tile.py --dataPath "D:\path\to\image\folder\"
```

___
### Stitch tiles in the Z dimension
script: `1_multiview_stitcher_3d.py`

**Motivation**: 
This pipeline is meant to stitch datasets that where split into multiple Substacks for Airyscan processing. 
This process requires a lot of RAM for processing, approximately 256 GB of RAM / 300 z planes. For this reason, datasets are split into smaller subsets for processing. If a file contains multiple tiles and mosaic data, each tile is processed separately in each run.
==Note==: this pipeline automatically converts images to `zarr`.

**Goal**:
Merge Substacks in 3D using `multiview-stitcher` python package.

**How**:
Run the `python` script in the command prompt of `conda` /`mamba`  with the proper environment. A mandatory argument of the path to the data folder is needed. An Additional arguments can be passed to define the image extension and, consequently, the package to open it. 
#TODO proof this approach for more file types

How to use:
1. open a terminal or, if in windows, the `conda`/`mamba` 
2. browse to the script directory using `cd` 
```shell
# in Linux or Mac
$ cd path/to/script/folder/

# in Windows
$ D: #if data is in drive D
$ cd path\to\script\folder\
```
3. then run the script and add the necessary arguments (see end of section for more information)
```bash
# if using Mac or Linux
python 1_multiview_stitcher_3d.py --dataPath "/path/to/image/folder/" --extension ".czi" --overlapTolerance 0
#in linux, use `tee` to save console output to file
python 1_multiview_stitcher_3d.py --dataPath "/path/to/image/folder/" --extension ".czi" --overlapTolerance 0 | tee /path/to/image/folder/log_name.txt

# if using Windows
python splitsave_czi_tile.czi.py --dataPath "D:\\path\\to\\image\\folder" --extension ".czi" --overlapTolerance 0
```

Arguments
```shell
--dataPath "/path/to/image/folder/" # the path to your data
--extension ".czi" # The extension of the files to be processed
--overlapTolerance # a tolerance overlap value, allowing stitching for more than the predefined value when there is sample drift
```

___
### Stitch tiles in 2d for full mosaic
script: `2_multiview_stitcher_2d.py`

**Motivation**: 
This pipeline is meant to stitch Mosaic datasets, merging and stitching tiles from an acquisition. 
This can be used after the previous script was run ([[#Stitching in 3D]]) or as a starting step.
_Note_: this pipeline automatically converts images to `zarr`  if reading from `czi` format. Otherwise, if the previous step was run, one has to choose the appropriate file extension and it will load existing `zarr` files.

**Goal**:
Merge Individual tile stacks from a mosaic using `multiview-stitcher` python package.

**How**:
Run the `python` script in the command prompt of `conda` /`mamba`  with the proper environment. A mandatory argument of the path to the data folder is needed (`--dataPath`). An Additional arguments can be passed to define the image extension and, consequently, the package to open it. 
#TODO proof this approach for more file types

How to use:
1. open a terminal or, if in windows, the `conda`/`mamba` 
2. browse to the script directory using `cd` 
```shell
# in Linux or Mac
$ cd path/to/script/folder/

# in Windows
$ D: #if data is in drive D
$ cd path\to\script\folder\
```
3. then run the script and add the necessary arguments (see end of section for more information)
```bash
# if using Mac or Linux
python 1_multiview_stitcher_3d.py --dataPath "/path/to/image/folder/" --extension ".zarr" --metadataSubstring "AcquisitionBlock" --tilePruningMethod "alternating_pattern" --overlapTolerance 0.15
#in linux, use `tee` to save console output to file
python 1_multiview_stitcher_3d.py --dataPath "/path/to/image/folder/" --extension ".zarr" --metadataSubstring "AcquisitionBlock"  --tilePruningMethod "alternating_pattern" --overlapTolerance 0.15 | tee /path/to/image/folder/log_name.txt

# if using Windows
python splitsave_czi_tile.czi.py --dataPath "D:\\path\\to\\image\\folder" --extension ".zarr" --metadataSubstring "AcquisitionBlock" --tilePruningMethod "alternating_pattern" --overlapTolerance 0.15
```
Arguments
```shell
--dataPath "/path/to/image/folder/" # the path to your data
--extension ".zarr" # The extension of the files to be processed
--metadataSubstring "_AcquisitioBlock" # the subtring from a czi file in a parent folder to look for metadata. Will fetch the stack position in metadata. i.e. name_AcquisitionBlock3 -> 3
--tilePruningMethod # manually define multiview-stitcher's tile pruning method
--overlapTolerance # a tolerance overlap value, allowing stitching for more than the predefined value when there is sample drift
```



## TO DO
- remove time (T) dimension from loading data in scripts for stitching. Image data should not have time dimension, if it has, probably it will break or overwrite data.