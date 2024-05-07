from .mnfinder import MNClassifier, MNModelDefaults
from .kerasmodels import AttentionUNet, MSAttentionUNet
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from skimage.filters import sobel
from skimage.measure import regionprops_table, label
from skimage.morphology import disk, opening
from skimage.exposure import rescale_intensity, adjust_gamma

class LaplaceDeconstruction(MNClassifier):
  """
  Laplace pyramids can separate an image into different frequencies, with each frequency 
  corresponding to a given level of informational detail.

  MN neural nets seem to rely heavily on examining the edges of nuclei to find associated MN.
  By breaking an image into a Laplacian pyramid and then recombining only the top 2 levels
  of detail, this removes information about the center of nuclei.

  This is an Attention UNet trained on these deconstructed images
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/LaplaceDeconstruction.tar.gz'

  crop_size = 128
  bg_max = 0.5
  fg_min = 0.1

  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)
    self.defaults.use_argmax = False
    self.defaults.opening_radius = 2

  def _get_mn_predictions(self, img):
    """
    Crops an image and generates a list of neural net predictions of each

    Parameters
    --------
    img : np.array
      The image to predict
    
    Returns
    --------
    list
      The coordinates of each crop in the original image in (r,c) format
    tf.Dataset
      The batched TensorFlow dataset used as input
    list
      The predictions
    """
    tensors = []
    coords = []
    num_channels = img.shape[2]
    crops = self._get_image_crops(img)

    sobel_idx = num_channels

    for crop in crops:
      lp = self._get_laplacian_pyramid(crop['image'][...,0], 2)
      new_img = lp[1]
      new_img = cv2.pyrUp(new_img, lp[0].shape[1::-1])
      new_img += lp[0]
      new_img += sobel(new_img)

      new_img = adjust_gamma(rescale_intensity(new_img, out_range=(0,1)), 2)

      tensors.append(tf.convert_to_tensor(
        np.expand_dims(new_img, axis=-1)
      ))
      coords.append(crop['coords'])

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset_batchs = dataset.batch(self.batch_size)
    predictions = self.trained_model.predict(dataset_batchs, verbose = 0)

    return coords, dataset, predictions

  def _get_needed_padding(self, img, num_levels):
    """
    Determine if a crop needs additional padding to generate 
    a Laplacian pyramid of a given depth

    Parameters
    --------
    img : np.array
      The image to predict
    num_levels : int
      The depth of the pyramid
    
    Returns
    --------
    int
      The needed x padding
    int
      The needed y padding
    """
    divisor = 2**num_levels

    x_remainder = img.shape[1]%divisor
    x_padding = (divisor-x_remainder) if x_remainder > 0 else 0

    y_remainder = img.shape[0]%divisor
    y_padding = (divisor-y_remainder) if y_remainder > 0 else 0

    return x_padding, y_padding

  def _pad_img(self, img, num_levels):
    """
    Pads a crop so that a Laplacian pyramid of a given depth
    can be made

    Parameters
    --------
    img : np.array
      The image to predict
    num_levels : int
      The depth of the pyramid
    
    Returns
    --------
    np.array
      The padded image
    """
    x_padding, y_padding = self._get_needed_padding(img, num_levels)
    if len(img.shape) == 2:
      new_img = np.zeros(( img.shape[0]+y_padding, img.shape[1]+x_padding), dtype=img.dtype)
    elif len(img.shape) == 3:
      new_img = np.zeros(( img.shape[0]+y_padding, img.shape[1]+x_padding, img.shape[2]), dtype=img.dtype)
    else:
      raise IncorrectDimensions()
    new_img[0:img.shape[0], 0:img.shape[1]] = img
    return new_img

  def _get_laplacian_pyramid(self, img, num_levels):
    """
    Builds a Laplacian pyramid of a given depth

    Parameters
    --------
    img : np.array
      The image to predict
    num_levels : int
      The depth of the pyramid
    
    Returns
    --------
    list
      List of levels
    """
    img = self._pad_img(img, num_levels)
    lp = []
    for i in range(num_levels-1):
      next_img = cv2.pyrDown(img)
      diff = img - cv2.pyrUp(next_img, img.shape[1::-1])
      lp.append(diff)
      img = next_img
    lp.append(img)

    return lp

  def _build_model(self):
    factory = AttentionUNet()
    return factory.build(self.crop_size, 1, 3)

  def _get_trainer(self, data_path, batch_size, num_per_image, augment=True):
    def post_process(data_points):
      for i in range(len(data_points)):
        lp = self._get_laplacian_pyramid(data_points[i]['image'][...,0], 2)
        new_img = lp[1]
        new_img = cv2.pyrUp(new_img, lp[0].shape[1::-1])
        new_img += lp[0]
        new_img += sobel(new_img)

        new_img = adjust_gamma(rescale_intensity(new_img, out_range=(0,1)), 2)
        new_img = np.expand_dims(new_img, axis=-1)

        data_points[i]['image'] = new_img

      return data_points
    return TFData(self.crop_size, data_path, batch_size, num_per_image, augment=augment, post_hooks=[ post_process ])

class Attention(MNClassifier):
  """
  A basic U-Net with additional attention modules in the decoder.

  Trained on single-channel images + Sobel
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Attention.tar.gz'

  crop_size = 128
  bg_max = 0.59
  fg_min = 0.24

  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)
    self.defaults.use_argmax = False

  def _build_model(self):
    factory = AttentionUNet()
    return factory.build(self.crop_size, 2, 3)

class Attention96(Attention):
  """
  A basic U-Net with additional attention modules in the decoder, but using a 96x96 crop size.

  Trained on single-channel images + Sobel
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Attention96.tar.gz'

  crop_size = 96
  bg_max = 0.59
  fg_min = 0.24

  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)

    self.defaults.use_argmax = True

class MSAttention(Attention):
  """
  An attention unet with an additional multi-scale modules on the front of each down block in the encoder.

  This performs convolutions at different resolutions and then runs MaxPooling on the concatenated results.

  Trained on single-channel images.
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/MSAttention.tar.gz'

  bg_max = 0.6
  fg_min = 0.3

  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)

    self.defaults.use_argmax = False
    self.defaults.opening_radius = 1

  def _get_mn_predictions(self, img):
    tensors = []
    coords = []
    num_channels = img.shape[2]
    crops = self._get_image_crops(img)

    sobel_idx = num_channels

    for crop in crops:
      tensors.append(tf.convert_to_tensor(
        np.expand_dims(crop['image'][...,0], axis=-1)
      ))
      coords.append(crop['coords'])

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset_batchs = dataset.batch(self.batch_size)
    predictions = self.trained_model.predict(dataset_batchs, verbose = 0)

    return coords, dataset, predictions

  def _build_model(self):
    factory = MSAttentionUNet()
    return factory.build(self.crop_size, 1, 3)

  def _get_trainer(self, data_path, batch_size, num_per_image, augment=True):
    def post_process(data_points):
      for i in range(len(data_points)):
        data_points[i]['image'] = np.expand_dims(data_points[i]['image'][...,0], axis=-1)

      return data_points
    return TFData(self.crop_size, data_path, batch_size, num_per_image, augment=augment, post_hooks=[ post_process ])

class MSAttention96(MSAttention):
  """
  A multi-scale attention UNet, but with 96x96 crop sizes.

  Trained on single-channel images.
  """

  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/MSAttention96.tar.gz'

  crop_size = 96
  bg_max = 0.6
  fg_min = 0.25

  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)

    self.defaults.use_argmax = True
    self.defaults.opening_radius = 1

# class SimpleCombined(MNClassifier):
#   """
#   A simple ensembling method where MN masks from multiple models
#   are combined together as a simple union, but with some size filtering
#   """
#   model_url = None

#   crop_size = 128

#   def __init__(self, weights_path=None, trained_model=None):
#     super().__init__() # There are no weights or trained model to load

#     # The base model will be used to generate
#     self.base_model = MNClassifier.get_model("Attention")
#     self.supplementary_models = [
#       MNClassifier.get_model("MSAttention")
#     ]
    
#   def _load_model(self, weights_path=None):
#     return True

#   def predict(self, img, skip_opening=None, expand_masks=None, use_argmax=None, area_thresh=250, **kwargs):
#     """
#     Generates MN and nuclear segments

#     Parameters
#     --------
#     img : np.array
#       The image to predict
#     skip_opening : bool|None
#       Whether to skip running binary opening on MN predictions. If None, defaults
#       to this model's value in self.defaults.skip_opening
#     expand_masks : bool|None
#       Whether to expand MN segments to their convex hull. If None, defaults
#       to self.defaults.expand_masks
#     use_argmax : bool|None
#       If true, pixel classes are assigned to whichever class has the highest
#       probability. If false, MN are assigned by self.bg_max and self.fg_min 
#       thresholds 
#     area_thresh : int|False
#       Larger MN that are separate from the nucleus tend to be called as nuclei.
#       Any nucleus segments < area_thresh will be converted to MN. If False, this
#       will not be done
    
#     Returns
#     --------
#     np.array
#       The nucleus labels
#     np.array
#       The MN labels
#     np.array
#       The raw output form the neural net
#     """
#     if skip_opening is None:
#       skip_opening = self.defaults.skip_opening

#     if expand_masks is None:
#       expand_masks = self.defaults.expand_masks

#     if use_argmax is None:
#       use_argmax = self.defaults.use_argmax

#     labels = self.base_model.predict(img, skip_opening, expand_masks, use_argmax, area_thresh)

#     base_mn_labels = (base_mn_labels != 0).astype(np.uint16)
#     for idx,model in enumerate(self.supplementary_models):
#       _, mn_labels, mn_nuc_labels, raw = model.predict(img, skip_opening, expand_masks, use_argmax, area_thresh)
#       mn_labels = opening(mn_labels, footprint=disk(2))
#       mn_nuc_labels = opening(mn_nuc_labels, footprint=disk(2))
#       mn_info = pd.DataFrame(regionprops_table(mn_labels, properties=('label', 'solidity', 'area')))
#       keep_labels = mn_info['label'].loc[(mn_info['area'] < 250)]
#       base_mn_labels[~np.isin(mn_labels, keep_labels)] = 0
#       base_mn_nuc_labels[~np.isin(mn_labels, keep_labels)] = 0

#     nucleus_labels[base_mn_labels != 0] = 0
    
#     return nucleus_labels, base_mn_labels, field_output
  
class Combined(MNClassifier):
  """
  An ensemble predictor

  Trained on the output of the Attention and MSAttention models
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Combined.tar.gz'
  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)

    self.crop_size = 128

    self.models = [
      MNClassifier.get_model("Attention"),
      MNClassifier.get_model("MSAttention")
    ]

    # self.model_url = None
    self.defaults.use_argmax = False
    self.bg_max = 0.6
    self.fg_min = 0.16

  def _get_mn_predictions(self, img):
    """
    Crops an image and generates a list of neural net predictions of each

    First gets the raw outputs of the models stored in self.models, then uses
    that as input to the model.

    Parameters
    --------
    img : np.array
      The image to predict
    
    Returns
    --------
    list
      The coordinates of each crop in the original image in (r,c) format
    tf.Dataset
      The batched TensorFlow dataset used as input
    list
      The predictions
    """
    tensors = []
    coords = []
    model_predictions = []

    for idx,model in enumerate(self.models):
      this_coords, dataset, predictions = model._get_mn_predictions(img)
      model_predictions.append(predictions)
      if len(coords) == 0:
        coords = this_coords

    num_crops = len(model_predictions[0])
    for crop_idx in range(num_crops):
      new_img = np.zeros((model_predictions[0][crop_idx].shape[0], model_predictions[0][crop_idx].shape[1], len(model_predictions)), dtype=np.float64)
      for model_idx in range(len(model_predictions)):
        new_img[...,model_idx] = model_predictions[model_idx][crop_idx][...,2].copy()
      tensors.append(tf.convert_to_tensor(new_img))

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset_batchs = dataset.batch(self.batch_size)
    predictions = self.trained_model.predict(dataset_batchs, verbose = 0)

    # Expand dims to match other models
    expanded = np.zeros((predictions.shape[0], predictions.shape[1], predictions.shape[2], 3), dtype=predictions.dtype)
    for idx,prediction in enumerate(predictions):
      expanded[idx][...,0] = np.min([ prediction[...,0], model_predictions[0][idx][...,0] ], axis=0)
      expanded[idx][...,1] = model_predictions[0][idx][...,1]
      expanded[idx][...,2] = prediction[...,1]

    return coords, dataset, expanded

  def _build_model(self):
    factory = AttentionUNet()
    return factory.build(self.crop_size, 2, 2)

  def _get_trainer(self, data_path, batch_size, num_per_image, augment=True):
    def post_process(data_points):
      for i in range(len(data_points)):
        channels = []
        for model in self.models:
          if model.name == 'Attention':
            _, _, mn_raw = model.predict(data_points[i]['image'])
          else:
            _, _, mn_raw = model.predict(np.expand_dims(data_points[i]['image'][...,0], axis=-1))

          channels.append(mn_raw)

        data_points[i]['image'] = np.stack(channels, axis=-1)

      return data_points
    return TFData(self.crop_size, data_path, batch_size, num_per_image, augment=augment, post_hooks=[ post_process ])