from brainextractor import BrainExtractor
import nibabel as nib
import simplebrainviewer as sbv
import cProfile

img = nib.load("data/T1_test.nii.gz")
bet = BrainExtractor(img)
bet.run()
# test = cProfile.run("bet.run()", sort="tottime")
mask = bet.compute_mask()
sbv.plot_brain(mask)
bet.save_mask("data/test.nii.gz")
