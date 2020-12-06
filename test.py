from brainextractor import BrainExtractor
import nibabel as nib

img = nib.load("data/T1_test.nii.gz")
bet = BrainExtractor(img)
bet.run(1000)
import simplebrainviewer as sbv
mask = bet.compute_mask()
sbv.plot_brain(mask)