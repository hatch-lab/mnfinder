from .mnfinder import MNClassifier, MNModelDefaults
from .kerasmodels import AttentionUNet, MSAttentionUNet, CombinedUNet
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from skimage.filters import sobel
from skimage.measure import regionprops_table, label
from skimage.morphology import disk, opening
from skimage.exposure import rescale_intensity, adjust_gamma

from pathlib import Path

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
    self.defaults.use_argmax = True

  def _build_model(self):
    factory = AttentionUNet()
    return factory.build(self.crop_size, 2, 3)

class MSAttention(MNClassifier):
  """
  A basic U-Net with additional attention modules in the decoder.

  Trained on single-channel images + Sobel
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/MSAttention.tar.gz'

  crop_size = 128
  bg_max = 0.59
  fg_min = 0.24

  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)
    self.defaults.use_argmax = True

  def _build_model(self):
    factory = MSAttentionUNet()
    return factory.build(self.crop_size, 2, 3)

class Combined(MNClassifier):
  """
  An ensemble predictor

  Trained on the output of the Attention and MSAttention models
  """
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Combined.tar.gz'
  def __init__(self, weights_path=None, trained_model=None):
    super().__init__(weights_path=weights_path, trained_model=trained_model)

    self.crop_size = 128

    # self.model_url = None
    self.defaults.use_argmax = True
    self.bg_max = 0.8
    self.fg_min = 0.05

  def _build_model(self):
    a = MNClassifier.get_model('Attention')
    base_model = a.trained_model
    base_model.trainable = False

    m = MNClassifier.get_model('MSAttention')
    adj_model = m.trained_model
    adj_model.trainable = False

    factory = CombinedUNet()
    return factory.build(base_model, adj_model, self.crop_size, 2)