# MN UNet segmenter
A package for segmenting micronuclei in micrographs.

## Quick-start
````
from mnfinder import MNModel
import numpy as np
from tifffile import TiffFile

trained_model = MNModel.get_model()

image = TiffFile.imread('path/to/image.tiff').asarray()
nuclei_labels, mn_labels, mn_raw = trained_model.predict(image)
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

MNFinder supports several different trained models with different architectures. The default is an Attention U-Net using 128x128 images as input.

Weights will be automatically downloaded.

### Available models
#### Attention
The default network is an Attention U-Net that was trained on 128x128 crops. 

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `False`
* `opening_radius`: 1

#### Attention96
An Attention U-Net trained on 96x96 crops.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `True`
* `opening_radius`: 2

#### MSAttention
This is a modification of the Attention U-Net that incorporates a multi-scale convolution in the down blocks.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `False`
* `opening_radius`: 1

#### MSAttention96
A multiscale Attention U-Net trained on 96x96 crops.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `True`
* `opening_radius`: 1

#### Combined
An Attention U-Net trained on the output of `Attention` and `MSAttention`.

**Defaults**
* `skip_opening`: `False`
* `expand_masks`: `True`
* `use_argmax`: `False`
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
nuclei_masks, mn_masks, mn_raw = trained_model.predict(img, skip_opening=[bool], expand_masks=[bool], use_argmax=[bool], area_thresh=[int])
````

A single method is used to predict nuclear and micronucler segments. This package is not designed to replace existing nuclear segmentation models--while it performs well at identifying nucleus pixel classes, it makes no effort at separating nuclei accurately.

These neural nets were trained on images taken at 20x. **Predictions for micrographs taken at other resolutions are greatly improved if they are scaled to match a 20x resolution.**

Images of arbitrary size will be cropped by a sliding window and segments combined.

#### Optional parameters
`skip_opening=bool`
: Whether to skip running opening on MN predictions prior to labelling. Many models are improved by removing small 1- or 2-px segments by image opening—erosion following by dilation. Defaults to the model default.

`expand_masks=bool`
: Whether to expand micronucleus masks by returning the convex hulls of each segment. Defaults to the model default.

`use_argmax=bool`
: Whether to determine pixel classes by taking the maximum probability. Some models are improved by instead setting a simple threshold on the micronucleus class probability, setting a pixel to the micronucleus class even if the model’s nucleus class probability is higher. If `use_argmax` is `False`, the model will select pixels with a background class > `model.bg_max` and a micronucleus class < `model.fg_min`. Defaults to the model default.

`area_thresh=int|False`
: Large micronuclei separated from the nucleus are often classed as nuclei. Any nucleus segments < `area_thresh` will be converted to micronuclei. Set to `False` to skip this conversion. Defaults to `250`.

### Training
````
model, model_history = trained_model.train(train_path=[path/to/training], val_path=[path/to/validation], batch_size=[int], epochs=[int], checkpoint_path=[path/to/training-checkpoints], num_per_image=[int], save_weights=[bool], save_path=[path/to/saved_model])
````

Each model was trained on a relatively small data set of RPE-1 and U2 OS cells. They may be improved by training on your own data set.

Training and validation data should be organized as follows:
````
path/to/folder/
- dataset1/
-- nucleus_masks/
-- mn_masks/
-- images/

- dataset2/
-- nucleus_masks/
-- mn_masks/
-- images/

...

- dataset_n/
-- nucleus_masks/
-- mn_masks/
-- images/
````

The name of the dataset folders is arbitrary and can be whatever you wish. The names of the images in the `images` folder is also arbitrary, but should match the names of the corresponding ground truth segments in the `nucleus_masks` and `mn_masks` folders.

`nucleus_masks` should contain images where nuclei have a pixel value > 0. `mn_masks` should likewise contain images where micronuclei have a pixel value > 0. If you wish to distinguish between intact and ruptured micronuclei, set intact micronuclei pixels to `1` and ruptured micronuclei pixels to `2`.

The following is an example set up for training and validation data:
````
path/to/training-data/
- imageset1/
-- nucleus_masks/
--- image1.png
--- image2.png

-- mn_masks/
--- image1.png
--- image2.png

-- images/
--- image1.tiff
--- image2.tiff

- imageset2/
-- nucleus_masks/
--- IMG2.tif
--- IMG3.tif

-- mn_masks/
--- IMG2.tif
--- IMG3.tif

-- images/
--- IMG2.tiff
--- IMG3.tiff

path/to/validation-data/
- imageset1/
-- nucleus_masks/
--- image3.png
--- image4.png

-- mn_masks/
--- image3.png
--- image4.png

-- images/
--- image3.tiff
--- image4.tiff
````

Images will be processed according to the needs of each model. If the images are multichannel, the first channel will be used for training.

Generally, images are cropped at random locations. Crops without any micronucleus or nucleus segments are ignored. Crops are then subject to augmentation via the Albumentations library.

#### Required parameters
`train_path=str|Path|None`
: Path to training data root. If None, will use this package's training data.

`val_path=str|Path|None`
: Path to validation data root. If None, will use this package's training data.

#### Optional parameters
`batch_size=int|None`
: Training batch size. If None, will default to the model’s prediction batch size. Defaults to `None`.

`epochs=int`
: The number of training epochs. Defaults to `100`.

`checkpoint_path=Path|str|None`
: Where to save checkpoints during training. If `None`, no checkpoints will be saved.
`num_per_image=int|None`
: The number of crops to return per image. Because crops are randomly positioned and can be randomly augmented, more crops can be extracted from a given image than otherwise. If `None`, will default to width/crop size * height/crop_size.

`save_weights=bool`
: Whether to save the model weights. Defaults to `True`.

`save_path=str|Path|None`
: Where to save model weights. If `None`, will overwrite the model’s default weights. Defaults to `None`.
