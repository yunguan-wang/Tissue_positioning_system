<img src="https://github.com/zzhu33/scSplitter/blob/master/QBRC.jpg">

# Tissue Positioning System (TPS) - a quantitative, unsupervised algorithm for zonated expression pattern detection in Immunofluorescence images.

## Welcome
* Tissue Positioning System algorithms is developed for learning zonated protein expression in hepatocytes, we welcome suggestions for other potential use cases.
* For current hepatocyte application, tps require an input IF image to have DAPI channel for nuclei and GS channel for central veins.
## Rationale / Motivation
* Protein expression in tissue is often in zonated patterns between/around certain feature.

* Manual identification of such patterns is not stably reproducible.

* Quantification of these zonated expression pattern very time consuming

## More tools
Researchers searching for more bioinformatics tools can visit our lab website: https://qbrc.swmed.edu/labs/wanglab/index.php.

## Requirement
TPS is developed and tested in Python3.6.

Installation requires [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/). [seaborn](https://github.com/mwaskom/seaborn), [scikit-image](https://scikit-image.org/) and [scikit-learn](https://scikit-learn.org/).

## Installation
```
git clone https://github.com/yunguan-wang/Tissue_positioning_system.git
cd Tissue_positioning_system
pip install . --upgrade
```
## Test installation
```
python scripts/worker_script.py input/example.tif -o output/example
```
```
Prosessing input/example.tif
Parameters: Namespace(dapi_cutoff=20, dapi_dilation_r=0, gs_higher_limit=0.75, gs_lower_limit=0.25, gs_step=0.1, input_img='input/example.tif', logging=False, maximal_neighbor_distance=20, output='', spot_size=False, tomato_cutoff=0, update=False, vessel_size_factor=2)
Segmentating using GS and DAPI
Merging neighboring masks...
Continue merging neighboring masks...
Continue merging neighboring masks...
Number of CV and PV: 28, 30
All opposite masks covered, stop expansion
All opposite masks covered, stop expansion
number of zones : 24
```
Outputs for the test example is in "./output/example/".

## Usage
tps can be executed easily with the command line worker script.

```
Worker script for tps

positional arguments:
  input_img             Absolute Input TIF image to be zonated, with signal of
                        interest at channel 0, GS at channel 1 and DAPI at
                        channel 2

optional arguments:
  -o, --output          output folder of results, if not supplied, it will be
                        that same as the input file name. (default: )
  -v, --vessel_size_factor
                        Vessel size threshold as x/10000 fold of image size
                        (default: 2)
  -d, --maximal_neighbor_distance
                        maximal pixel distance between two neighboring masks
                        to be considered as two separate masks. (default: 20)
  -c, --dapi_cutoff
                        Dapi cutoff value for hard thresholding. (default: 20)
  -gl, --gs_lower_limit
                        The lower percentatge limit of GS signal intensity
                        within a mask, which is used in classify CV from PV
                        (default: 0.25)
  -gh, --gs_higher_limit
                        The higher percentatge limit of GS signal intensity
                        within a mask, which is used in classify CV from PV
                        (default: 0.75)
  -gs, --gs_step
                        The interval of percentage in the GS intensity
                        features. (default: 0.1)
```
## Gallery
Example results

<img src="https://github.com/yunguan-wang/liver_zone_segmentation/blob/biohpc/output/example/example_results.png" height="450" width="800">

## Contact
If you have any questions or suggestions, please contact Yunguan Wang (yunguan.wang@utsouthwestern.edu) or Guanghua Xiao (guanghua.xiao@utsouthwestern.edu).
