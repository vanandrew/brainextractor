#!/usr/bin/env python

from brainextractor import BrainExtractor
import nibabel as nib
import simplebrainviewer as sbv

img = nib.load("data/t1w_data/sub-MSC03_ses-struct01_run-01_T1w.nii.gz")
bet = BrainExtractor(img)
bet.run(iterations=1000)
mask = bet.compute_mask()
sbv.plot_brain(mask)
bet.save_mask("data/test.nii.gz")
