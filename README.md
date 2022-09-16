# brainextractor
A reimplementation of FSL's Brain Extraction Tool in Python.

Follows the algorithm as described in:

```
Smith SM. Fast robust automated brain extraction. Hum Brain Mapp.
2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568; PMCID: PMC6871816.
```

https://user-images.githubusercontent.com/3641187/190677589-be019bc6-60e4-4e96-8c71-266285ab0755.mp4

## Install

To install, simply use `pip` to install this repo:

```
# install from pypi
pip install brainextractor

# install repo with pip
pip install git+https://github.com/vanandrew/brainextractor@main

# install from local copy
pip install /path/to/local/repo
```

Note that it is recommended to use `brainextractor` on python 3.7+

## Usage

To extract a brain mask from a image, you can call:

```
# basic usage
brainextractor [input_image] [output_image]

# example
brainextractor /path/to/test_image.nii.gz /path/to/some_output_image.nii.gz
```

You can adjust the fractional intensity with the `-f` flag:

```
# with custom set threshold
brainextractor [input_image] [output_image] -f [threshold]

# example
brainextractor /path/to/test_image.nii.gz /path/to/some_output_image.nii.gz -f 0.4
```

To view the deformation process, you can use the `-w` flag to write the
surfaces to a file. Then use `brainextractor_render` to view them:

```
# writes surfaces to file
brainextractor [input_image] [output_image] -w [surfaces_file]

# load surfaces and render
brainextractor_render [surfaces_file]

# example
brainextractor /path/to/test_image.nii.gz /path/to/some_output_image.nii.gz -w /path/to/surface_file.surfaces

brainextractor_render /path/to/surface_file.surfaces
```

If you need an explanation of the options at any time, simply run the help:

```
brainextractor --help
```

If you need to call Brainextractor directly from python:
```python
# import the nibabel library so we can read in a nifti image
import nibabel as nib
# import the BrainExtractor class
from brainextractor import BrainExtractor

# read in the image file first
input_img = nib.load("/content/MNI.nii.gz")

# create a BrainExtractor object using the input_img as input
# we just use the default arguments here, but look at the
# BrainExtractor class in the code for the full argument list
bet = BrainExtractor(img=input_img)

# run the brain extraction
# this will by default run for 1000 iterations
# I recommend looking at the run method to see how it works
bet.run()

# save the computed mask out to file
bet.save_mask("/content/MNI_mask.nii.gz")
```
