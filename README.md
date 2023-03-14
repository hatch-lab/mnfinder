# MN UNet segmenter
A package for segmenting micronuclei in micrographs.

## Usage
````
from MNUNet import MNModel
import numpy as np
from tifffile import TiffFile

trained_model = MNModel.get_model('FocalLossCombined')

image = TiffFile.imread('path/to/image.tiff').asarray()
iamge = np.stack([ image ], axis=-1)
nuclei_labels, mn_labels, mn_raw = trained_model.predict_field(image)
````