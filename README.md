# MN UNet segmenter
A package for segmenting micronuclei in micrographs.

## Usage
````
from mnfinder import MNModel
import numpy as np
from tifffile import TiffFile

trained_model = MNModel.get_model('Combined')

image = TiffFile.imread('path/to/image.tiff').asarray()
nuclei_labels, mn_labels, mn_raw = trained_model.predict_field(image)
````