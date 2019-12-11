# Game of Zones (GoZ) - a quantitative, unsupervised algorithm for zonated expression pattern detection in Immunofluorescence images.

## Welcome
* Game of Zones algorithms is developed for learning zonated protein expression in hepatocytes, we welcome suggestions for other potential use cases.
* For current hepatocyte application, GoZ require an input IF image to have DAPI channel for nuclei and GS channel for central veins.
## Rationale / Motivation
* Protein expression in tissue is often in zonated patterns between/around certain feature.

* Manual identification of such patterns is not stably reproducible.

* Quantification of these zonated expression pattern very time consuming

## Requirement
GoZ is developed and tested in Python3.6.

Installation requires [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/). [seaborn](https://github.com/mwaskom/seaborn), [scikit-image](https://scikit-image.org/) and [scikit-learn](https://scikit-learn.org/).

## Install
```
git clone https://github.com/yunguan-wang/liver_zone_segmentation.git
cd liver_zone_segmentation`
pip install -e . --upgrade`
```
## Test installation
```
python test.py
```
```
Processing test example in "./input/example.tif"...
Prosessing ./input/example.tif
Segmentating using GS and DAPI
Merging neighboring masks...
Continue merging neighboring masks...
```
Outputs for the test example is in "./output/example/".

## Usage
GoZ can be executed easily with the command line worker script.

```
Worker script for GoZ

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
![Example image](https://github.com/yunguan-wang/liver_zone_segmentation/blob/master/input/example.JPG)

![Segmented image with CV(brown) and PV(white)](https://github.com/yunguan-wang/liver_zone_segmentation/blob/master/output/example/Marker%20segmented_classfied.png)

![Predicted lobules](https://github.com/yunguan-wang/liver_zone_segmentation/blob/master/output/example/lobules.JPG)

![Marker expression status by zones](https://github.com/yunguan-wang/liver_zone_segmentation/blob/master/output/example/Marker%20signal%20intensity%20in%20zones.png)