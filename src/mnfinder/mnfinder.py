from csbdeep.utils import normalize
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as K
import json
from skimage.filters import sobel
from skimage.measure import regionprops_table, label
from skimage.morphology import disk, binary_opening, opening, binary_erosion, area_opening, binary_closing, convex_hull_image
from skimage.exposure import rescale_intensity, adjust_gamma
from skimage.color import label2rgb
from skimage.segmentation import clear_border
import pandas as pd
import numpy as np
import cv2
from tifffile import TiffWriter, TiffFile
import requests
import tarfile
from tqdm import tqdm
import math

import matplotlib.pyplot as plt

class MNModelDefaults:
  """
  Class for storing model defaults

  This allows for easy overriding of default parameters in individual classifier models.

  It's basically an overcomplicated dictionary
  """
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      self.__dict__[key] = value

  def __setattr__(self, name, value):
    self.__dict__[f"{name}"] = value

  def __getattr__(self, name):
    return self.__dict__[f"{name}"]

class MNModel:
  """
    Base class for a MN segmenter.

    Attributes
    ----------
    models_root : Path
      Where model files are stored
    crop_size : int
      The input width and height of the model
    oversample_size : int
      The amount of overlap between crops when scanning across an image
    batch_size : int
      Batch size for running predictions
    name : str
      The name of this model
    bg_max : float
      If not using argmax to decide pixel classes, the maximum threshold
      a pixel can have for class 0 and still be considered a MN (class 2)
    fg_min : float
      If not using argmax to decide pixel classes, the minimum threshold
      a pixel can have for class 2 and still be considered a MN
    defaults : MNModelDefaults
      Stores defaults for prediction parameters. Typically includes:
      skip_opening : bool
        Whether to skip performing binary opening on predictions
      opening_radius : int
        The radius of a disk used as the footprint for binary_opening
      expand_masks : bool
        Whether to return the convex hulls of MN segments
      use_argmax : bool
        Whether to assign pixel classes by argmax, or by thresholds
    model_url : str
      The URL for downloading model weights


    Static methods
    --------
    get_available_models()
      Return the names of all available predictors
    is_model_available(model_name=str)
      Whether the given model name exists
    get_model(model_name=str)
      Returns an instance of the predictor with the given name
    normalize_image(img=np.array)
      Normalizes the intensity and data type of an image
    normalize_dimensions(img=np.array)
      Normalizes the image shape
    eval_mn_prediction(mn_true_masks=np.array, mn_labels=np.array)
      Generates metrics on how well a prediction is performing given
      a ground-truth mask


    Public methods
    --------
    predict(img=np.array, skip_opening=bool|None, area_thresh=int, **kwargs)
      Generates masks of nuclei and micronuclei for a given image
    """

  models_root = (Path(__file__) / "../../../models").resolve()

  @staticmethod
  def get_available_models():
    """
    Return the list of available model classes

    Static method

    Returns
    --------
    list
    """
    available_models = [ 
      'Combined',
      'SimpleCombined',
      'Attention',
      'Attention96',
      'MSAttention',
      'MSAttention96',
      'LaplaceDeconstruction'
    ]
    return available_models
  
  @staticmethod
  def is_model_available(model_name):
    """
    Checks if a given model class exists

    Static method

    Parameters
    --------
    model_name : str
      The model name
    
    Returns
    --------
    bool
    """
    return model_name in MNModel.get_available_models()

  @staticmethod
  def get_model(model_name='Attention'):
    """
    Returns an instance of the given model

    Static method

    Parameters
    --------
    model_name : str
      The model name. Defaults to the Attention class
    
    Returns
    --------
    MNModel
    """
    available_models = MNModel.get_available_models()
    if model_name not in available_models:
      raise ModelNotFound("No such MN model: {}".format(model_name))
    try:
      model = globals()[model_name]
      return model()
    except:
      raise ModelNotLoaded("Could not load model: {}".format(model_name))

  @staticmethod
  def normalize_image(img):
    """
    Normalizes the intensity and datatype of an image

    Static method

    Parameters
    --------
    img : np.array
      The image
    
    Returns
    --------
    np.array
      Scaled using cbsdeep.normalize, and converted to np.float64
    """
    return normalize(img, 4, 99, dtype=np.float64)

  @staticmethod
  def normalize_dimensions(img):
    """
    Normalizes image shape

    2D images are reshaped to (height, width, 1)

    Static method

    Parameters
    --------
    img : np.array
      The image
    
    Returns
    --------
    np.array
      Scaled using cbsdeep.normalize, and converted to np.float64
    """
    if len(img.shape) == 3:
      return img
    if len(img.shape) == 2:
      return np.stack([ img ], axis=-1)
    raise IncorrectDimensions()

  @staticmethod
  def eval_mn_prediction(mn_true_masks, mn_labels):
    """
    Evaluates the results of a prediction against ground truth

    Generates pd.DataFrames with various metrics

    Static method

    Parameters
    --------
    mn_true_masks : np.array
      Ground truth. MN masks should be annotated such that pixels
      belonging to a given MN have connectivity=1 (Rook-style 
      contiguity) and separate MN have at least 1 px of space or
      only touch at the diagonal

      If so desired, ruptured MN can be assigned a pixel value of 2
      and intact a pixel value of 1, and this will be included in the
      analysis. Often, ruptured MN are more difficult to identify,
      likely because smaller MN are more likely to rupture
    mn_labels : np.array
      The predicted labels
      
    Returns
    --------
    pd.DataFrame
      Information about each true MN segment
    pd.DataFrame
      Information about each prediction
    pd.DataFrame
      Summary statistics
    """
    true_mn_labels = label(mn_true_masks, connectivity=1)

    intact_mn = np.zeros_like(true_mn_labels)
    ruptured_mn = np.zeros_like(true_mn_labels)
    intact_mn[(mn_true_masks == 1)] = 1
    ruptured_mn[(mn_true_masks == 2)] = 1

    mn_df = {
      'true_mn_label': [], # The real MN label
      'intact': [], # If this MN is intact or ruptured
      'found': [], # If any portion of this segment overlapped with 1 or more predictions
      'area': [], # The area in square pixels
      'proportion_segmented': [], # The amount of overlap between prediction and truth
      'pred_labels': [], # The label IDs of any predictions that overlap
    }

    pred_df = {
      'pred_mn_label': [], # The prediction label
      'exists': [], # If any portion of this prediction overlapped with 1 or more real MN
      'area': [], # The area in square pixels
      'overlap_proportion': [], # The proportion of overlap between prediction and truths
      'true_labels': [], # The label IDs of any true MN that overlap
    }

    summary_df = {
      'num_mn': [], # The number of MN in this image
      'num_intact_mn': [], # The number of intact MN
      'num_ruptured_mn': [], # The number of ruptured MN
      'num_predictions': [], # The number of predictions
      'num_mn_found': [], # The number of MN that overlap to any degree with predictions
      'num_intact_mn_found': [], # The number of intact MN that overlap to any degree with predictions
      'num_ruptured_mn_found': [], # The number of ruptured MN that overlap to any degree with predictions
      'iou': [], # The overall intersection over union of this image
      'intersection': [], # The intersection of predictions and truth
      'divergence': [] # The proportion of predictions that do not overlap with truth
    }
    # Summary also contains PPV and and recall statistics

    for mn_label in np.unique(true_mn_labels):
      if mn_label == 0:
        continue

      mn_df['true_mn_label'].append(mn_label)
      if np.sum(intact_mn[(true_mn_labels == mn_label)]) > 0:
        mn_df['intact'].append(True)
      else:
        mn_df['intact'].append(False)

      area = np.sum(true_mn_labels[(true_mn_labels == mn_label)])
      mn_df['area'].append(area)

      if np.sum(mn_labels[(true_mn_labels == mn_label)]) > 0:
        mn_df['found'].append(True)
        mn_df['proportion_segmented'].append(
          np.sum(mn_labels[(true_mn_labels == mn_label)])/area
        )
        mn_df['pred_labels'].append(np.unique(mn_labels[(true_mn_labels == mn_label) & (~(mn_labels == 0))]))
      else:
        mn_df['found'].append(False)
        mn_df['proportion_segmented'].append(0.)
        mn_df['pred_labels'].append([])

    for mn_label in np.unique(mn_labels):
      pred_df['pred_mn_label'].append(mn_label)
      area = np.sum(mn_labels[(mn_labels == mn_label)])
      pred_df['area'].append(area)

      if np.sum(true_mn_labels[(mn_labels == mn_label)]) > 0:
        pred_df['exists'].append(True)
        mn_df['overlap_proportion'].append(
          np.sum(true_mn_labels[(mn_labels == mn_label)])/area
        )
        mn_df['true_labels'].append(np.unique(true_mn_labels[(mn_labels == mn_label) & (~(true_mn_labels == 0))]))
      else:
        pred_df['exists'].append(False)
        mn_df['overlap_proportion'].append(0.)
        mn_df['true_labels'].append([])

    mn_df = pd.DataFrame(mn_df)
    pred_df = pd.DataFrame(pred_df)

    summary_df['num_mn'].append(mn_df.shape[0])
    summary_df['num_intact_mn'].append(np.sum(mn_df['intact']))
    summary_df['num_ruptured_mn'].append(mn_df.shape[0]-np.sum(mn_df['intact']))
    summary_df['num_predictions'].append(pred_df.shape[0])
    summary_df['num_mn_found'].append(np.sum(mn_df['found']))
    summary_df['num_intact_mn_found'].append(np.sum(mn_df.loc[mn_df['intact'] == True, 'found']))
    summary_df['num_ruptured_mn_found'].append(np.sum(mn_df.loc[mn_df['intact'] == False, 'found']))

    intersection = np.sum(np.logical_and((mn_labels > 0), (true_mn_labels > 0)))
    union = np.sum(np.logical_or((mn_labels > 0), (true_mn_labels > 0)))
    if union == 0:
      summary_df['iou'].append(0)
    else:
      summary_df['iou'].append(intersection / union)
    summary_df['intersection'].append(intersection)
    summary_df['divergence'].append(np.sum(mn_labels > 0)-intersection)

    summary_df = pd.DataFrame(summary_df)

    summary_df['ppv'] = summary_df['num_mn_found']/summary_df['num_predictions']
    summary_df['recall'] = summary_df['num_mn_found']/summary_df['num_mn']
    summary_df['intact_recall'] = summary_df['num_intact_mn_found']/summary_df['num_intact_mn']
    summary_df['ruptured_recall'] = summary_df['num_ruptured_mn_found']/summary_df['num_ruptured_mn']

    return mn_df, pred_df, summary_df

  def __init__(self):
    """
    Constructor
    
    Returns
    --------
    MNModel
    """
    self.crop_size = 96
    self.oversample_size = self.crop_size//4
    self.batch_size = 64
    self.name = self.__class__.__name__
    self.bg_max = 0.5
    self.fg_min = 0.2

    self.defaults = MNModelDefaults(
      skip_opening=False, 
      expand_masks=True, 
      use_argmax=True, 
      opening_radius=1
    )

    self._load_model()

  def _load_model(self):
    """
    Load the trained model weights

    If the model weights have not yet been downloaded, will fetch the tar.gz
    and unpack the files
    """
    model_path = self.models_root / self.__class__.__name__
    model_gzip_path = self.models_root / (self.__class__.__name__ + ".tar.gz")
    if not model_path.exists():
      # Try to download
      r = requests.get(self.model_url, allow_redirects=True, stream=True)

      if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError("Could not fetch model")

      total_size = int(int(r.headers.get('Content-Length', 0))/(1024*1024))
      with open(model_gzip_path, 'wb') as f:
        pbar = tqdm(total=total_size, desc="Fetching " + self.__class__.__name__, unit="MiB", bar_format='{l_bar}{bar}|{n:0.2f}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}{postfix}]')
        for chunk in r.iter_content(chunk_size=8192):
          if chunk:
            f.write(chunk)
            pbar.update(len(chunk)/(1024*1024))

      pbar.close()

      print('Unpacking...')
      with tarfile.open(model_gzip_path) as f:
        f.extractall(self.models_root)

      model_gzip_path.unlink()

    self.trained_model = tf.keras.models.load_model(
      str(self._get_path()), 
      compile=False, 
      custom_objects=self._get_custom_metrics()
    )
  
  def predict(self, img, skip_opening=None, expand_masks=None, use_argmax=None, area_thresh=250, **kwargs):
    """
    Generates MN and nuclear segments

    Parameters
    --------
    img : np.array
      The image to predict
    skip_opening : bool|None
      Whether to skip running binary opening on MN predictions. If None, defaults
      to this model's value in self.defaults.skip_opening
    expand_masks : bool|None
      Whether to expand MN segments to their convex hull. If None, defaults
      to self.defaults.expand_masks
    use_argmax : bool|None
      If true, pixel classes are assigned to whichever class has the highest
      probability. If false, MN are assigned by self.bg_max and self.fg_min 
      thresholds 
    area_thresh : int|False
      Larger MN that are separate from the nucleus tend to be called as nuclei.
      Any nucleus segments < area_thresh will be converted to MN. If False, this
      will not be done
    
    Returns
    --------
    np.array
      The nucleus labels
    np.array
      The MN labels
    np.array
      The raw output form the neural net
    """
    if skip_opening is None:
      skip_opening = self.defaults.skip_opening

    if expand_masks is None:
      expand_masks = self.defaults.expand_masks

    if use_argmax is None:
      use_argmax = self.defaults.use_argmax

    img = self.normalize_image(self.normalize_dimensions(img))
    if img.shape[0] < self.crop_size or img.shape[1] < self.crop_size:
      raise ValueError("Image is smaller than minimum size of {}x{}".format(self.crop_size, self.crop_size))

    coords, dataset, predictions = self._get_mn_predictions(img)
    num_channels = predictions[0].shape[2]
    field_output = np.zeros(( img.shape[0], img.shape[1], num_channels ), dtype=np.float64)

    for idx, batch in enumerate(dataset):
      field_output = self._blend_crop(field_output, predictions[idx], coords[idx])

    field_labels = np.argmax(field_output, axis=-1).astype(np.uint8)
    nucleus_labels = (field_lables == 1).astype(np.uint8)
    if use_argmax:
      mn_labels = (field_labels == 2).astype(np.uint8)
    else:
      mn_labels = ((field_output[...,0] < self.bg_max) & (field_output[...,2] > self.fg_min)).astype(np.uint8)

    nucleus_labels = clear_border(nucleus_labels)
    nucleus_labels = label(nucleus_labels)

    if area_thresh is not False and area_thresh > 0:
      possible_mn_info = pd.DataFrame(regionprops_table(nucleus_labels, properties=('label', 'area')))
      switch_labels = mn_info['label'].loc[(mn_info['area'] < area_thresh)]
      nucleus_labels[np.isin(nucleus_labels, switch_labels)] = 0
      mn_labels[np.isin(nucleus_labels, switch_labels)] = 1

    if not skip_opening:
      mn_labels = binary_opening(mn_labels, footprint=disk(self.defaults.opening_footprint)).astype(np.uint8)
    mn_labels = clear_border(mn_labels)
    mn_labels = label(mn_labels, connectivity=1)

    if expand_masks and len(np.unique(mn_labels)) > 1:
      mn_labels = self._expand_masks(mn_labels)

    nucleus_labels[mn_labels > 0] = 0

    return nucleus_labels, mn_labels, field_output

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

    for crop in crops:
      tensors.append(tf.convert_to_tensor(
        np.stack([ crop['image'][...,0], crop['image'][...,1] ], axis=-1)
      ))
      coords.append(crop['coords'])

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset_batchs = dataset.batch(self.batch_size)
    predictions = self.trained_model.predict(dataset_batchs, verbose = 0)

    return coords, dataset, predictions

  def _get_image_crops(self, img):
    """
    Generates crops of an image

    Each crop will have 2*n channels, containing Sobel
    edge detection run on each channel independently

    Parameters
    --------
    img : np.array
      The image to predict
    
    Returns
    --------
    list
      A list of dictionaries containing the crop and coordinates
    """
    channels = []
    edges = []
    for channel in range(img.shape[2]):
      channels.append(self.normalize_image(img[...,channel]))

    edges = [ sobel(x) for x in channels ]
    edges = [ self.normalize_image(x) for x in edges ]

    channels += edges

    return self._get_sliding_window_crops(channels)

  def _get_sliding_window_crops(self, channels):
    """
    Generates crops of an image by sliding window

    Parameters
    --------
    channels : list
      A list of individual channels + Sobel channels
    
    Returns
    --------
    list
      A list of dictionaries containing the crop and coordinates
    """
    width = channels[0].shape[1]
    height = channels[0].shape[0]

    crops = []

    slide_px = self.crop_size-self.oversample_size

    this_y = 0
    while(this_y <= height):
      this_x = 0
      while(this_x <= width):
        crop = np.zeros(( self.crop_size, self.crop_size, len(channels) ), dtype=channels[0].dtype)

        left = this_x
        right = left + self.crop_size
        top = this_y
        bottom = top + self.crop_size

        if right > width:
          right = width
        if bottom > height:
          bottom = height

        for idx,channel in enumerate(channels):
          crop_width = right-left
          crop_height = bottom-top
          crop[0:crop_height,0:crop_width,idx] = channel[top:bottom, left:right]

        crops.append({
          'image': crop,
          'coords': (left, right, top, bottom )
        })

        this_x += slide_px
      this_y += slide_px

    return crops

  def _blend_crop(self, field, crop, coords):
    """
    Blend crops together using linear blending

    This method is designed to be called iteratively,
    with each crop added to an existing field,
    which is then modified and can be used as input for
    the next iteration

    Parameters
    --------
    field : np.array
      The output from the last time _blend_crop was called
    crop : np.array
      The prediction
    coords : list
      The coordinates where this crop should be placed
      
    Returns
    --------
    np.array
      The modified field with the crop blended in
    """
    left   = coords[0]
    right  = coords[1]
    top    = coords[2]
    bottom = coords[3]
    
    # Merge images together
    mask = np.ones(( self.crop_size, self.crop_size ), np.float64)
    # Top feather
    if top > 0:
      mask[0:self.oversample_size, :] = np.tile(np.linspace(0,1,self.oversample_size), (self.crop_size,1)).T
    # Bottom feather
    if bottom < field.shape[0]:
      mask[self.crop_size-self.oversample_size:self.crop_size, :] = np.tile(np.linspace(1,0,self.oversample_size), (self.crop_size,1)).T
    # Left feather
    if left > 0:
      mask[:, 0:self.oversample_size] = np.tile(np.linspace(0,1,self.oversample_size), (self.crop_size, 1))
    # Right feather
    if right < field.shape[1]:
      mask[:, self.crop_size-self.oversample_size:self.crop_size] = np.tile(np.linspace(1,0,self.oversample_size), (self.crop_size, 1))

    # Top-left
    if top > 0 and left > 0:
      mask[0:self.oversample_size, 0:self.oversample_size] = np.tile(np.linspace(0,1,self.oversample_size), (self.oversample_size,1)).T*np.tile(np.linspace(0,1,self.oversample_size), (self.oversample_size, 1))
    # Top-right
    if top > 0 and right < field.shape[1]:
      mask[0:self.oversample_size, self.crop_size-self.oversample_size:self.crop_size] = np.fliplr(np.tile(np.linspace(0,1,self.oversample_size), (self.oversample_size,1)).T*np.tile(np.linspace(0,1,self.oversample_size), (self.oversample_size, 1)))
    # Bottom-left
    if bottom < field.shape[0] and left > 0:
      mask[self.crop_size-self.oversample_size:self.crop_size, 0:self.oversample_size] = np.fliplr(np.tile(np.linspace(1,0,self.oversample_size), (self.oversample_size,1)).T*np.tile(np.linspace(1,0,self.oversample_size), (self.oversample_size, 1)))
    # Bottom-right
    if bottom < field.shape[0] and right < field.shape[1]:
      mask[self.crop_size-self.oversample_size:self.crop_size, self.crop_size-self.oversample_size:self.crop_size] = np.tile(np.linspace(1,0,self.oversample_size), (self.oversample_size,1)).T*np.tile(np.linspace(1,0,self.oversample_size), (self.oversample_size, 1))

    for c_idx in range(field.shape[2]):
      crop[...,c_idx] *= mask

    field[top:bottom, left:right] += crop[0:bottom-top, 0:right-left]

    return field

  def _expand_masks(self, mn_labels):
    """
    Returns the convex hulls of mn_labels

    Parameters
    --------
    mn_labels : np.array
      The labeled segmentation results
      
    Returns
    --------
    np.array
      The modified labels
    """
    new_field_labels = np.zeros_like(field_labels)
    for field_label in np.unique(field_labels):
      if field_label == 0:
        continue
      img_copy = np.zeros_like(field_labels, dtype=bool)
      img_copy[field_labels == field_label] = True
      img_copy = convex_hull_image(img_copy)
      new_field_labels[img_copy] = field_label

    return new_field_labels

  @staticmethod
  def _get_model_metric(name):
    """
    Returns custom model metrics

    Needed for loading trained models and avoiding warnings about custom metrics
    not being loaded

    Parameters
    --------
    name : str
      The metric to return
      
    Returns
    --------
    fun
      The function
    """
    def sigmoid_focal_crossentropy(y_true, y_pred, alpha = 0.25, gamma = 2.0, from_logits = False,):
      """
      Implements the focal loss function.
      
      Parameters
      --------
      y_true : tensor
        True targets
      y_pred : tensor
        Predictions
      alpha : float
        Balancing factor
      gamma : float
        Modulating factor
      from_logits : bool
        Passed to binary_crossentropy()
        
      Returns
      --------
      tensor
        Weighted loss float tensor
      """

      def _safe_mean(losses, num_present):
        """
        Computes a safe mean of the losses.

        Parameters
        --------
        losses : tensor
          Individual loss measurements
        num_present : int
          The number of measurable elements
        
        Returns
        --------
        float
          Mean of losses unless num_present == 0, in which case 0 is returned
        """
        total_loss = tf.reduce_sum(losses)
        return tf.math.divide_no_nan(total_loss, num_present, name="value")

      def _num_elements(losses):
        """
        Computes the number of elements in losses tensor

        Parameters
        --------
        losses : tensor
          Individual loss measurements
        
        Returns
        --------
        int
          The number of elements
        """
        with K.name_scope("num_elements") as scope:
          return tf.cast(tf.size(losses, name=scope), dtype=losses.dtype)

      if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

      y_pred = tf.convert_to_tensor(y_pred)
      y_true = tf.cast(K.one_hot(tf.cast(y_true, tf.uint8), num_classes=4), dtype=y_pred.dtype)

      # Get the cross_entropy for each entry
      ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

      # If logits are provided then convert the predictions into probabilities
      if from_logits:
        pred_prob = tf.sigmoid(y_pred)
      else:
        pred_prob = y_pred

      p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
      alpha_factor = 1.0
      modulating_factor = 1.0

      if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

      if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

      # compute the final loss and return
      # tf.print(tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1))
      losses = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
      loss = _safe_mean(losses, _num_elements(losses))

      return loss

    def sigmoid_focal_crossentropy_loss(y_true, y_pred):
      """
      Wrapper for sigmoid_focal_crossentropy
      """
      return sigmoid_focal_crossentropy(y_true, y_pred)

    def dice_coef(y_true, y_pred, smooth=1):
      """
      Calculates the Sørensen–Dice coefficient
      
      Parameters
      --------
      y_true : tensor
        True targets
      y_pred : tensor
        Predictions
      smooth : float
        Smoothing factor
        
      Returns
      --------
      tensor
      """
      y_true_f = tf.cast(K.flatten(K.one_hot(tf.cast(y_true, dtype=tf.uint8), num_classes=3)[...,1:]), dtype=tf.float32)
      y_pred_f = K.flatten(tf.cast(y_pred[...,1:], dtype=tf.float32))
      intersection = K.sum(y_true_f * y_pred_f, axis=-1)
      denom = K.sum(y_true_f + y_pred_f, axis=-1)
      return K.mean((2. * intersection / (denom + smooth)))

    def mean_iou(y_true, y_pred, smooth=1):
      """
      Calculates the mean IOU of just the MN segmentation
      
      Parameters
      --------
      y_true : tensor
        True targets
      y_pred : tensor
        Predictions
      smooth : float
        Smoothing factor
        
      Returns
      --------
      tensor
      """
      y_true_f = tf.cast(K.flatten(K.one_hot(tf.cast(y_true, dtype=tf.uint8), num_classes=3)[...,2]), dtype=tf.float32)
      y_pred_f = K.flatten(tf.cast(y_pred[...,2:4], dtype=tf.float32))
      intersection = K.sum(y_true_f * y_pred_f, axis=-1)
      union = K.sum(y_true_f + y_pred_f, axis=-1)-intersection
      return (intersection + smooth)/(union + smooth)
    
    def mean_iou_with_nuc(y_true, y_pred, smooth=1):
      """
      Calculates the mean IOU of both MN and nucleus segmentation
      
      Parameters
      --------
      y_true : tensor
        True targets
      y_pred : tensor
        Predictions
      smooth : float
        Smoothing factor
        
      Returns
      --------
      tensor
      """
      y_true_f = tf.cast(K.flatten(K.one_hot(tf.cast(y_true, dtype=tf.uint8), num_classes=3)[...,1:3]), dtype=tf.float32)
      y_pred_f = K.flatten(tf.cast(y_pred[...,1:3], dtype=tf.float32))
      intersection = K.sum(y_true_f * y_pred_f, axis=-1)
      union = K.sum(y_true_f + y_pred_f, axis=-1)-intersection
      return (intersection + smooth)/(union + smooth)

    metrics = {
      'sigmoid_focal_crossentropy': sigmoid_focal_crossentropy,
      'sigmoid_focal_crossentropy_loss': sigmoid_focal_crossentropy_loss,
      'dice_coef': dice_coef,
      'mean_iou': mean_iou,
      'mean_iou_with_nuc': mean_iou_with_nuc
    }

    return metrics[name]

  def _get_custom_metrics(self):
    """
    Returns the custom model metrics for this class

    Needed for loading trained models and avoiding warnings about custom metrics
    not being loaded

    Returns
    --------
    dict
      Dictionary of custom metric names and their associated functions
    """
    return { 
      'sigmoid_focal_crossentropy_loss': self._get_model_metric('sigmoid_focal_crossentropy_loss'), 
      'sigmoid_focal_crossentropy': self._get_model_metric('sigmoid_focal_crossentropy'),
      'mean_iou': self._get_model_metric('mean_iou'),
      'mean_iou_with_nuc': self._get_model_metric('mean_iou_with_nuc')
    }

  def _get_path(self):
    """
    Get the root path of this model

    Returns
    --------
    Path
    """
    return MNModel.models_root / self.name


class LaplaceDeconstruction(MNModel):
  """
  Laplace pyramids can separate an image into different frequencies, with each frequency 
  corresponding to a given level of informational detail.

  MN neural nets seem to rely heavily on examining the edges of nuclei to find associated MN.
  By breaking an image into a Laplacian pyramid and then recombining only the top 2 levels
  of detail, this removes information about the center of nuclei.

  This is an Attention UNet trained on these deconstructed images
  """
  def __init__(self):
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/LaplaceDeconstruction.tar.gz'

    super().__init__()

    self.crop_size = 128
    self.bg_max = 0.5
    self.fg_min = 0.1
   
    self.defaults.use_argmax = False
    self.defaults.opening_footprint = 2

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

class Attention(MNModel):
  """
  A basic U-Net with additional attention modules in the decoder.

  Trained on single-channel images + Sobel
  """
  def __init__(self):
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Attention.tar.gz'

    super().__init__()

    self.crop_size = 128

    self.defaults.use_argmax = False

    self.bg_max = 0.59
    self.fg_min = 0.24

class Attention96(MNModel):
  """
  A basic U-Net with additional attention modules in the decoder, but using a 96x96 crop size.

  Trained on single-channel images + Sobel
  """
  def __init__(self):
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Attention96.tar.gz'

    super().__init__()
    self.defaults.use_argmax = True

    self.bg_max = 0.59
    self.fg_min = 0.24

class MSAttention(Attention):
  """
  An attention unit with an additional multi-scale modules on the front of each down block in the encoder.

  This performs convolutions at different resolutions and then runs MaxPooling on the concatenated results.

  Trained on single-channel images.
  """
  def __init__(self):
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/MSAttention.tar.gz'

    super().__init__()

    self.defaults.use_argmax = False

    self.bg_max = 0.6
    self.fg_min = 0.3
    self.defaults.opening_footprint = 2

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

class MSAttention96(MSAttention):
  """
  A multi-scale attention UNet, but with 96x96 crop sizes.

  Trained on single-channel images.
  """
  def __init__(self):
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/MSAttention96.tar.gz'

    super().__init__()

    self.defaults.use_argmax = True

    self.crop_size = 96
    self.bg_max = 0.6
    self.fg_min = 0.25
    self.defaults.opening_footprint = 1

class SimpleCombined(MNModel):
  """
  A simple ensembling method where MN masks from multiple models
  are combined together as a simple union, but with some size filtering
  """
  def __init__(self):
    super().__init__()

    self.crop_size = 128

    # The base model will be used to generate
    self.base_model = MNModel.get_model("Attention")
    self.supplementary_models = [
      MNModel.get_model("MSAttention")
    ]

    self.model_url = None
    self.defaults.privilege_nuclei = False
    
  def _load_model(self):
    return True

  def predict(self, img, skip_opening=None, expand_masks=None, use_argmax=None, area_thresh=250, **kwargs):
    """
    Generates MN and nuclear segments

    Parameters
    --------
    img : np.array
      The image to predict
    skip_opening : bool|None
      Whether to skip running binary opening on MN predictions. If None, defaults
      to this model's value in self.defaults.skip_opening
    expand_masks : bool|None
      Whether to expand MN segments to their convex hull. If None, defaults
      to self.defaults.expand_masks
    use_argmax : bool|None
      If true, pixel classes are assigned to whichever class has the highest
      probability. If false, MN are assigned by self.bg_max and self.fg_min 
      thresholds 
    area_thresh : int|False
      Larger MN that are separate from the nucleus tend to be called as nuclei.
      Any nucleus segments < area_thresh will be converted to MN. If False, this
      will not be done
    
    Returns
    --------
    np.array
      The nucleus labels
    np.array
      The MN labels
    np.array
      The raw output form the neural net
    """
    if skip_opening is None:
      skip_opening = self.defaults.skip_opening

    if expand_masks is None:
      expand_masks = self.defaults.expand_masks

    if use_argmax is None:
      use_argmax = self.defaults.use_argmax

    nucleus_labels, base_mn_labels, field_output = self.base_model.predict(img, skip_opening, expand_masks, use_argmax, area_thresh)

    base_mn_labels = (base_mn_labels != 0).astype(np.uint16)
    for idx,model in enumerate(self.supplementary_models):
      _, mn_labels, mn_raw = model.predict(img, skip_opening, expand_masks, use_argmax, area_thresh)
      mn_labels = opening(mn_labels, footprint=disk(2))
      mn_info = pd.DataFrame(regionprops_table(mn_labels, properties=('label', 'solidity', 'area')))
      keep_labels = mn_info['label'].loc[(mn_info['area'] < 250)]
      base_mn_labels[np.isin(mn_labels, keep_labels)] = 1
      field_output += mn_raw

    nucleus_labels[base_mn_labels != 0] = 0
    base_mn_labels = label(base_mn_labels, connectivity=1)
    
    return nucleus_labels, base_mn_labels, field_output
  

class Combined(MNModel):
  """
  An ensemble predictor

  Trained on the output of the Attention and MSAttention models
  """
  def __init__(self):
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/Combined.tar.gz'

    super().__init__()

    self.crop_size = 128

    self.models = [
      MNModel.get_model("Attention"),
      MNModel.get_model("MSAttention")
    ]

    # self.model_url = None
    self.defaults.use_argmax = False
    self.bg_max = 0.55
    self.fg_min = 0.27

  # def _load_model(self):
  #   return True

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

class IncorrectDimensions(Exception):
  "Images must be (x,y,c) or (x,y)"
  pass

class ModelNotFound(Exception):
  "That model could not be found"
  pass

class ModelNotLoaded(Exception):
  "That model could not be loaded"
  pass