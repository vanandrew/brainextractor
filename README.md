# brainextractor
A reimplementation of FSL's Brain Extraction Tool in Python

## Install

## Usage

```
usage: brainextractor [-h] [-w WRITE_SURFACE_DEFORM] [-f FRACTIONAL_THRESHOLD] [-n ITERATIONS]
                      [-t HISTOGRAM_THRESHOLD HISTOGRAM_THRESHOLD] [-d SEARCH_DISTANCE SEARCH_DISTANCE]
                      [-r RADIUS_OF_CURVATURES RADIUS_OF_CURVATURES]
                      input_img output_img

A Reimplementation of FSL's Brain Extraction Tool

positional arguments:
  input_img             Input image to brain extract
  output_img            Output image to write out

optional arguments:
  -h, --help            show this help message and exit
  -w WRITE_SURFACE_DEFORM, --write_surface_deform WRITE_SURFACE_DEFORM
                        Path to write out surface files at each deformation step
  -f FRACTIONAL_THRESHOLD, --fractional_threshold FRACTIONAL_THRESHOLD
                        Main threshold parameter for controlling brain/background (Default: 0.5)
  -n ITERATIONS, --iterations ITERATIONS
                        Number of iterations to run (Default: 1000)
  -t HISTOGRAM_THRESHOLD HISTOGRAM_THRESHOLD, --histogram_threshold HISTOGRAM_THRESHOLD HISTOGRAM_THRESHOLD
                        Sets min/max of histogram (Default: 0.02, 0.98)
  -d SEARCH_DISTANCE SEARCH_DISTANCE, --search_distance SEARCH_DISTANCE SEARCH_DISTANCE
                        Sets search distance for max/min of image along vertex normals (Default: 20.0, 10.0)
  -r RADIUS_OF_CURVATURES RADIUS_OF_CURVATURES, --radius_of_curvatures RADIUS_OF_CURVATURES RADIUS_OF_CURVATURES
                        Sets min/max radius of curvature for surface (Default: 3.33, 10.0)

Author: Andrew Van, vanandrew@wustl.edu, 12/15/2020
```

```
usage: brainextractor_render [-h] [-s SAVE_MP4] [-l] surfaces

Renders surface deformation evolution

positional arguments:
  surfaces              Directory to display surfaces

optional arguments:
  -h, --help            show this help message and exit
  -s SAVE_MP4, --save_mp4 SAVE_MP4
                        Saves an mp4 output
  -l, --loop            Loop the render (1 hour)

Author: Andrew Van, vanandrew@wustl.edu, 12/15/2020
```