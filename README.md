# MN UNet segmenter
A package for segmenting micronuclei in micrographs.

## Quick-start
````
from mnfinder import MNClassifier
import numpy as np
from tifffile import TiffFile

trained_model = MNClassifier.get_model()

image = TiffFile.imread('path/to/image.tiff').asarray()
labels = trained_model.predict(image)
````

## Installation
MNFinder depends on TensorFlow. It will be installed for you via `pip`.

### pip
````
pip install mnfinder
````

## Usage
### Loading a model
````
trained_model = MNModel.get_model([model_name])
````

MNFinder supports several different trained models with different architectures. The default is an Attention U-Net.

Weights will be automatically downloaded.

### Available models
#### Attention
The default network is an Attention U-Net that was trained on 128x128 crops. 

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `True`
* `opening_radius`: 1

#### MSAttention
This is a modification of the Attention U-Net that incorporates a multi-scale convolution in the down blocks.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `True`
* `opening_radius`: 1

#### Combined
An Attention U-Net trained on the micronucleus output of `Attention` and `MSAttention`.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `True`
* `opening_radius`: 1

#### LaplaceDeconstruction
Images are first transformed into Laplace pyramids, and recombined only using the top 2 levels of the pyramid to highlight cell edges.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `False`
* `opening_radius`: 2

### Predictions
````
img = np.array(Image.open("my/image.png"))
labels = trained_model.predict(img, skip_opening=[bool], expand_masks=[bool], use_argmax=[bool], area_thresh=[int], return_raw_output=[bool])
````
A single method is used to predict and label nuclear and micronucler segments. 

These neural nets were trained on images taken at 20x. **Predictions for micrographs taken at other resolutions are greatly improved if they are scaled to match a 20x resolution.**

Images of arbitrary size will be cropped by a sliding window and segments combined.

Labels are returned as a 3-channel image. Channel 1 contains the unique cell label for each segmented nucleus; channel 2 has each micronucleus labelled with its corresponding cell label; channel 3 has a unique label for each micronucleus. 

#### Optional parameters
`skip_opening=bool`
: Whether to skip running opening on MN predictions prior to labelling. Many models are improved by removing small 1- or 2-px segments by image opening—erosion following by dilation. Defaults to the model default.

`expand_masks=bool`
: Whether to expand micronucleus masks by returning the convex hulls of each segment. Defaults to the model default.

`use_argmax=bool`
: Whether to determine pixel classes by taking the maximum probability. Some models are improved by instead setting a simple threshold on the micronucleus class probability, setting a pixel to the micronucleus class even if the model’s nucleus class probability is higher. If `use_argmax` is `False`, the model will select pixels with a background class > `model.bg_max` and a micronucleus class < `model.fg_min`. Defaults to the model default.

`area_thresh=int|False`
: Large micronuclei separated from the nucleus are often classed as nuclei. Any nucleus segments < `area_thresh` will be converted to micronuclei. Set this to `False` to skip this conversion. Defaults to `250`.

`return_raw_output=bool`
: If you wish to examine the fields returned by the semantic and instance classifier neural nets, set this to `True`.

### Prediction info
````
mn_df, nuc_df = MNClassifier.get_label_data(labels)
````
Provides some basic information about each MN and nuclear label predicted.