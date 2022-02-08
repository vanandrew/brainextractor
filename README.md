# brainextractor
A reimplementation of FSL's Brain Extraction Tool in Python.

Follows the algorithm as described in:

```
Smith SM. Fast robust automated brain extraction. Hum Brain Mapp.
2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568; PMCID: PMC6871816.
```

## Install

To install, simply use `pip` to install this repo:

```
# install from pypispheresphere
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
