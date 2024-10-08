import sys
from csbdeep.utils import normalize
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from skimage.filters import sobel, threshold_li, threshold_yen
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table, label
from skimage.morphology import disk, binary_opening, convex_hull_image, skeletonize, binary_dilation
from skimage.exposure import rescale_intensity, adjust_gamma, adjust_sigmoid
from skimage.segmentation import clear_border, watershed, find_boundaries, expand_labels
import pandas as pd
import numpy as np
import cv2
from tifffile import tifffile
import requests
import tarfile
from tqdm import tqdm
import math
import warnings
import random
from PIL import Image
import albumentations as A
from datetime import datetime
from platformdirs import PlatformDirs
import inspect
import importlib
from importlib.metadata import version

from cdBoundary.boundary import ConcaveHull
from scipy.ndimage import distance_transform_edt
from scipy import spatial

__version__ = "1.2.3"
__model_version__ = "1.2.1"
dirs = PlatformDirs("MNFinder", "Hatch-Lab", __model_version__)
Path(dirs.user_data_dir).parent.mkdir(exist_ok=True)
Path(dirs.user_data_dir).mkdir(exist_ok=True)

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
  training_root : Path
    Where training data is stored
  testing_root : Path
    Where testing data is stored
  crop_size : int
    The input width and height of the model
  oversample_ratio : float
    The amount of overlap between crops when scanning across an image, as a proportion of crop_size
  batch_size : int
    Batch size for running predictions
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
  train(train_root=Path|str, val_root=Path|str, batch_size=None|int, epochs=100, checkpoint_path=Path|str|None, num_per_image=int|None)
    Build and train a model from scratch
  """
  models_root = (Path(dirs.user_data_dir) / "models").resolve()
  training_root = (Path(__file__) / "../training-data").resolve()
  validation_root = (Path(__file__) / "../validation-data").resolve()
  testing_root = (Path(__file__) / "../test-data").resolve()

  class_type = "none"

  def get_tf_version(self):
    tf_version = version('tensorflow').split('.')
    return int(tf_version[0]), int(tf_version[1])

  @staticmethod
  def get_available_models():
    """
    Return the list of available model classes

    Static method

    Returns
    --------
    list
    """
    raise MethodNotImplemented("Ambiguous calling from MNModel. Call from MNClassifier or MNSegmenter.")
  
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
    raise MethodNotImplemented("Ambiguous calling from MNModel. Call from MNClassifier or MNSegmenter.")

  @staticmethod
  def get_model(model_name='Attention', weights_path=None, trained_model=None):
    """
    Returns an instance of the given model

    Static method

    Parameters
    --------
    model_name : str
      The model name. Defaults to the Attention class
    weights_path : Path|str|None
      Where to load the weights. If None, will load pretrained weights
    trained_model : tf.keras.Model|None
      To substitute an existing neural net model, specify it here
    
    Returns
    --------
    MNModel
    """
    raise MethodNotImplemented("Ambiguous calling from MNModel. Call from MNClassifier or MNSegmenter.")

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
    # return normalize(adjust_sigmoid(img, cutoff=0.5, gain=5), 4, 99, dtype=np.float64)
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
  def get_label_data(labels, nuc_img=None, mn_img=None, additional_metrics=None):
    """
    Generates pd.DataFrames with various metrics on the predicted labels

    Static method

    Parameters
    --------
    labels : np.array
      The predicted labels from MNClassifier.predict()
    nuc_img : np.array|None
      Image to use for intensity measurements for nuclei
    mn_img : np.array|None
      Image to use for intensity measurements for MN
    additional_metrics : list|None
      Additional metrics to request from skimage.measure.regionprops_table

    Returns
    --------
    pd.DataFrame
      Information about each MN prediction
    pd.DataFrame
      Information about each nucleus prediction
    """
    properties = ['label', 'centroid', 'area']

    if additional_metrics is not None and len(additional_metrics) > 0:
      properties = np.unique(np.concatenate([ properties, additional_metrics ]))

    # Pred stats
    try:
      pred_nuc_df = pd.DataFrame(regionprops_table(labels[...,0], nuc_img, properties=properties)).rename(
        columns={ 
          'label': 'cell_label', 
          'centroid-0': 'y',
          'centroid-1': 'x'
        }
      )
    except AttributeError as e:
      if "unavailable when `intensity_image` has not been specified" in str(e):
        raise AttributeError("`nuc_img` is required")
      else:
        raise e

    pred_mn_df = []
    for cell_label in pred_nuc_df['cell_label'].unique():
      this_cell_x = pred_nuc_df.loc[pred_nuc_df['cell_label'] == cell_label, 'x'].iloc[0]
      this_cell_y = pred_nuc_df.loc[pred_nuc_df['cell_label'] == cell_label, 'y'].iloc[0]

      this_mn_labels = labels[...,2].copy()
      this_mn_labels[labels[...,1] != cell_label] = 0
      try:
        this_mn_stats = pd.DataFrame(regionprops_table(this_mn_labels, mn_img, properties=properties)).rename(
          columns={ 
            'label': 'mn_label', 
            'centroid-0': 'y',
            'centroid-1': 'x'
          }
        )
      except AttributeError as e:
        if "unavailable when `intensity_image` has not been specified" in str(e):
          raise AttributeError("`mn_img` is required")
        else:
          raise e

      this_mn_stats['cell_label'] = cell_label
      this_mn_stats['distance_to_nuc'] = ((this_mn_stats['x']-this_cell_x)**2+(this_mn_stats['y']-this_cell_y)**2)**(1/2)
      pred_mn_df.append(this_mn_stats)
    pred_mn_df = pd.concat(pred_mn_df)

    # Put cell label, mn label front
    mn_cols = list(pred_mn_df.columns)
    mn_cols.insert(0, mn_cols.pop(mn_cols.index('cell_label')))
    mn_cols.insert(0, mn_cols.pop(mn_cols.index('mn_label')))

    cell_cols = list(pred_nuc_df.columns) 
    cell_cols.insert(0, cell_cols.pop(cell_cols.index('cell_label')))

    pred_mn_df = pred_mn_df.loc[:, mn_cols]
    pred_nuc_df = pred_nuc_df.loc[:, cell_cols]

    return pred_mn_df, pred_nuc_df

  @staticmethod
  def eval_mn_prediction(full_mask, labels, nuc_img=None, mn_img=None, additional_metrics=None):
    """
    Evaluates the results of a prediction against ground truth

    Generates pd.DataFrames with various metrics

    Static method

    Parameters
    --------
    full_mask : np.array
      Ground truth. Should be an NxNx3 matrix. All pixels that are
      nuclei should be [ 1, ID, 0 ]. All pixels that are MN should
      be [ 2, ID, MN_ID ], where ID is a unique ID for each cell
      and MN_ID is a unique ID for each MN. 

      If so desired, ruptured MN can be assigned with pixel values of
      [ 3, ID, MN_ID ], and this will be included in the
      analysis. Often, ruptured MN are more difficult to identify,
      likely because smaller MN are more likely to rupture
    labels : np.array
      The predicted labels from MNClassifier.predict()
    nuc_img : np.array|None
      Image to use for intensity measurements for nuclei
    mn_img : np.array|None
      Image to use for intensity measurements for MN
    additional_metrics : list|None
      Additional metrics to request from skimage.measure.regionprops_table
      
    Returns
    --------
    pd.DataFrame
      Information about each true MN segment
    pd.DataFrame
      Information about each true nucleus segment
    pd.DataFrame
      Information about each MN prediction
    pd.DataFrame
      Information about each nucleus prediction
    pd.DataFrame
      Summary statistics
    """
    intact_mn = np.zeros(( full_mask.shape[0], full_mask.shape[1] ), dtype=np.uint16)
    ruptured_mn = np.zeros(( full_mask.shape[0], full_mask.shape[1] ), dtype=np.uint16)
    intact_mn[(full_mask[...,0] == 2)] = 1
    ruptured_mn[(full_mask[...,0] == 3)] = 1

    summary_df = {
      'num_mn': [], # The number of MN in this image
      'num_nuclei': [],
      'num_intact_mn': [], # The number of intact MN
      'num_ruptured_mn': [], # The number of ruptured MN
      'num_mn_predictions': [], # The number of predictions
      'num_nuclei_predictions': [], # The number of predictions
      'num_mn_found': [], # The number of MN that overlap to any degree with predictions
      'num_nuclei_found': [], 
      'num_intact_mn_found': [], # The number of intact MN that overlap to any degree with predictions
      'num_ruptured_mn_found': [], # The number of ruptured MN that overlap to any degree with predictions
      'mn_iou': [], # The overall intersection over union of this image
      'nuclei_iou': [],
      'combined_iou': [],
      'mn_intersection': [], # The intersection of predictions and truth
      'nuclei_intersection': [], 
      'combined_intersection': [],
      'mn_divergence': [], # The proportion of predictions that do not overlap with truth
      'nuclei_divergence': [],
      'combined_divergence': []
    }
    # Summary also contains PPV and and recall statistics

    # True stats
    true_labels = np.zeros_like(full_mask)
    true_labels[...,0] = clear_border(full_mask[...,1]).copy()
    true_labels[...,1] = clear_border(full_mask[...,1]).copy()
    true_labels[...,2] = clear_border(full_mask[...,2]).copy()

    true_labels[...,0][full_mask[...,2] != 0] = 0 # Clear out MN
    true_labels[...,1][full_mask[...,2] == 0] = 0 # Clear out Nuc

    true_mn_df, true_nuc_df = MNClassifier.get_label_data(true_labels, nuc_img, mn_img, additional_metrics)
    pred_mn_df, pred_nuc_df = MNClassifier.get_label_data(labels, nuc_img, mn_img, additional_metrics)

    # Extra info to add
    true_mn_df['intact'] = True # If this MN is intact or ruptured
    true_mn_df['found'] = False # If any portion of this segment overlapped with 1 or more predictions
    true_mn_df['proportion_segmented'] = 0.0 # The amount of overlap between prediction and truth
    true_mn_df['pred_labels'] = "" # The label IDs of any predictions that overlap
    true_mn_df['pred_cell_labels'] = ""

    pred_mn_df['exists'] = False # If any portion of this prediction overlapped with 1 or more real MN
    pred_mn_df['proportion_true'] = 0.0 # The proportion of overlap between prediction and truths
    pred_mn_df['true_ids'] = "" # The label IDs of any true MN that overlap
    pred_mn_df['correctly_assigned'] = False # If this MN is assigned to the correct nucleus

    true_nuc_df['found'] = False
    true_nuc_df['proportion_segmented'] = 0.0 
    true_nuc_df['pred_labels'] = ""

    pred_nuc_df['exists'] = False
    pred_nuc_df['proportion_true'] = 0.0
    pred_nuc_df['true_ids'] = ""

    for cell_id in true_nuc_df['cell_label'].unique():
      mn_ids = true_mn_df.loc[true_mn_df['cell_label'] == cell_id, 'mn_label'].unique()
      for mn_id in mn_ids:
        idx = (full_mask[...,2] == mn_id)
        if np.sum(intact_mn[idx]) > 0:
          true_mn_df.loc[true_mn_df['mn_label'] == mn_id, 'intact'] = True

        pred_overlap = np.sum(np.logical_and(idx, labels[...,2] > 0))
        if pred_overlap > 0:
          true_mn_df.loc[true_mn_df['mn_label'] == mn_id, 'found'] = True
          true_mn_df.loc[true_mn_df['mn_label'] == mn_id, 'proportion_segmented'] = pred_overlap/np.sum(idx)
          true_mn_df.loc[true_mn_df['mn_label'] == mn_id, 'pred_labels'] = ",".join(np.unique(labels[(idx) & (labels[...,2] != 0),2]).astype(str))
          true_mn_df.loc[true_mn_df['mn_label'] == mn_id, 'pred_cell_labels'] = ",".join(np.unique(labels[(idx) & (labels[...,1] != 0),1]).astype(str))
      
      idx = (full_mask[...,1] == cell_id)
      pred_overlap = np.sum(np.logical_and(idx, labels[...,0] > 0))
      if pred_overlap > 0:
        true_nuc_df.loc[true_nuc_df['cell_label'] == cell_id, 'found'] = True
        true_nuc_df.loc[true_nuc_df['cell_label'] == cell_id, 'proportion_segmented'] = pred_overlap/np.sum(idx)
        true_nuc_df.loc[true_nuc_df['cell_label'] == cell_id, 'pred_labels'] = ",".join(np.unique(labels[(idx) & (labels[...,0] != 0),0]).astype(str))

    for cell_id in pred_nuc_df['cell_label'].unique():
      mn_ids = pred_mn_df.loc[pred_mn_df['cell_label'] == cell_id, 'mn_label'].unique()
      for mn_id in mn_ids:
        idx = (labels[...,2] == mn_id)

        pred_overlap = np.sum(np.logical_and(idx, full_mask[...,2] > 0))
        if pred_overlap > 0:
          pred_mn_df.loc[pred_mn_df['mn_label'] == mn_id, 'exists'] = True
          pred_mn_df.loc[pred_mn_df['mn_label'] == mn_id, 'proportion_true'] = pred_overlap/np.sum(idx)
          pred_mn_df.loc[pred_mn_df['mn_label'] == mn_id, 'true_ids'] = ",".join(np.unique(full_mask[(idx) & (full_mask[...,2] != 0),2]).astype(str))

          cell_ids = np.unique(full_mask[(idx) & (full_mask[...,2] != 0),1])
          if np.intersect1d(full_mask[(labels[...,0] == cell_id),1], cell_ids).shape[0] > 0:
            pred_mn_df.loc[pred_mn_df['mn_label'] == mn_id, 'correctly_assigned'] = True

      idx = (labels[...,0] == cell_id)
      pred_overlap = np.sum(np.logical_and(idx, full_mask[...,0] > 0))
      if pred_overlap > 0:
        pred_nuc_df.loc[pred_nuc_df['cell_label'] == cell_id, 'exists'] = True
        pred_nuc_df.loc[pred_nuc_df['cell_label'] == cell_id, 'proportion_true'] = pred_overlap/np.sum(idx)
        pred_nuc_df.loc[pred_nuc_df['cell_label'] == cell_id, 'true_ids'] = ",".join(np.unique(full_mask[(idx) & (full_mask[...,1] != 0),1]).astype(str))

    summary_df['num_mn'].append(true_mn_df.shape[0])
    summary_df['num_nuclei'].append(true_nuc_df.shape[0])
    summary_df['num_intact_mn'].append(np.sum(true_mn_df['intact']))
    summary_df['num_ruptured_mn'].append(true_mn_df.shape[0]-np.sum(true_mn_df['intact']))
    summary_df['num_mn_predictions'].append(pred_mn_df.shape[0])
    summary_df['num_nuclei_predictions'].append(pred_nuc_df.shape[0])
    summary_df['num_mn_found'].append(np.sum(true_mn_df['found']))
    summary_df['num_nuclei_found'].append(np.sum(true_nuc_df['found']))
    summary_df['num_intact_mn_found'].append(np.sum(true_mn_df.loc[true_mn_df['intact'] == True, 'found']))
    summary_df['num_ruptured_mn_found'].append(np.sum(true_mn_df.loc[true_mn_df['intact'] == False, 'found']))

    mn_intersection = np.sum(np.logical_and((full_mask[...,2] > 0), (labels[...,2] > 0)))
    mn_union = np.sum(np.logical_or((full_mask[...,2] > 0), (labels[...,2] > 0)))
    if mn_union == 0:
      summary_df['mn_iou'].append(0)
    else:
      summary_df['mn_iou'].append(mn_intersection / mn_union)
    summary_df['mn_intersection'].append(mn_intersection)
    summary_df['mn_divergence'].append(np.sum(labels[...,2] > 0)-mn_intersection)

    nuc_intersection = np.sum(np.logical_and((full_mask[...,0] > 0), (labels[...,0] > 0)))
    nuc_union = np.sum(np.logical_or((full_mask[...,0] > 0), (labels[...,0] > 0)))
    if nuc_union == 0:
      summary_df['nuclei_iou'].append(0)
    else:
      summary_df['nuclei_iou'].append(nuc_intersection / nuc_union)
    summary_df['nuclei_intersection'].append(nuc_intersection)
    summary_df['nuclei_divergence'].append(np.sum(labels[...,0] > 0)-nuc_intersection)

    combined_intersection = mn_intersection + nuc_intersection
    combined_union = mn_union + nuc_union
    if combined_union == 0:
      summary_df['combined_iou'].append(0)
    else:
      summary_df['combined_iou'].append(combined_intersection / combined_union)
    summary_df['combined_intersection'].append(combined_intersection)
    summary_df['combined_divergence'].append(np.sum((labels[...,2] > 0) | (labels[...,0] > 0))-combined_intersection)

    summary_df = pd.DataFrame(summary_df)

    summary_df['mn_ppv'] = summary_df['num_mn_found']/summary_df['num_mn_predictions']
    summary_df['nuclei_ppv'] = summary_df['num_nuclei_found']/summary_df['num_nuclei_predictions']
    summary_df['mn_recall'] = summary_df['num_mn_found']/summary_df['num_mn']
    summary_df['nuclei_recall'] = summary_df['num_nuclei_found']/summary_df['num_nuclei']
    summary_df['intact_recall'] = summary_df['num_intact_mn_found']/summary_df['num_intact_mn']
    summary_df['ruptured_recall'] = summary_df['num_ruptured_mn_found']/summary_df['num_ruptured_mn']

    return true_mn_df, true_nuc_df, pred_mn_df, pred_nuc_df, summary_df

  crop_size = 128
  oversample_ratio = 0.25
  batch_size = 64
  bg_max = 0.5
  fg_min = 0.2

  def __init__(self, weights_path=None, trained_model=None):
    """
    Constructor

    Parameters
    --------
    weights_path : str|Path|None
      Where the model weights are stored. If None, defaults to models/[model_name]
    trained_model : tf.keras.Model|None
      If we wish to supply your own trained model, otherwise it will be loaded
    """
    self.defaults = MNModelDefaults(
      skip_opening=False, 
      expand_masks=True, 
      use_argmax=True, 
      opening_radius=1
    )

    if trained_model is not None:
      self.trained_model = trained_model
    else:
      self._load_model(weights_path)

  def _load_model(self, weights_path=None):
    """
    Load the trained model weights

    If the model weights have not yet been downloaded, will fetch the tar.gz
    and unpack the files

    Parameters
    --------
    weights_path : str|Path|None
      Where the model weights are stored. If None, defaults to models/[model_name]/final.weights.h5
    """
    if weights_path is None:
      weights_path = self._get_path() / "final.weights.h5"
    else:
      weights_path = Path(weights_path).resolve()

    model_gzip_path = self.models_root / (type(self).__name__ + ".tar.gz")
    if not weights_path.exists() and not (weights_path.parent / (weights_path.name + ".index")).exists():
      # Try to download
      r = requests.get(self.model_url, allow_redirects=True, stream=True)

      if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError("Could not fetch model")

      total_size = int(int(r.headers.get('Content-Length', 0))/(1024*1024))
      with open(model_gzip_path, 'wb') as f:
        pbar = tqdm(total=total_size, desc="Fetching " + type(self).__name__, unit="MiB", bar_format='{l_bar}{bar}|{n:0.2f}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}{postfix}]')
        for chunk in r.iter_content(chunk_size=8192):
          if chunk:
            f.write(chunk)
            pbar.update(len(chunk)/(1024*1024))

      pbar.close()

      print('Unpacking...')
      with tarfile.open(model_gzip_path) as f:
        f.extractall(self.models_root)

      model_gzip_path.unlink()

    self.trained_model = self._build_model()
    if self.get_tf_version()[1] < 16:
      self.trained_model.load_weights(weights_path, skip_mismatch=True, by_name=True)
    else:
      self.trained_model.load_weights(weights_path, skip_mismatch=True)

  def _get_field_predictions(self, img):
    coords, dataset, predictions = self._get_mn_predictions(img)
    num_channels = predictions[0].shape[2]
    field_output = np.zeros(( img.shape[0], img.shape[1], num_channels ), dtype=np.float64)

    for idx in range(len(dataset)):
      field_output = self._blend_crop(field_output, predictions[idx], coords[idx])

    return field_output

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
    channels = [ self.normalize_image(img[...,0]) ]
    edges = []
    # for channel in range(img.shape[2]):
    #   channels.append(self.normalize_image(img[...,channel]))

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

    oversample_size = int(self.crop_size*self.oversample_ratio)

    slide_px = self.crop_size-oversample_size

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
    # Don't want to modify this in-place
    crop = crop.copy()
    
    left   = coords[0]
    right  = coords[1]
    top    = coords[2]
    bottom = coords[3]

    oversample_size = int(self.crop_size*self.oversample_ratio)
    
    # Merge images together
    mask = np.ones(( self.crop_size, self.crop_size ), np.float64)
    # Top feather
    if top > 0:
      mask[0:oversample_size, :] = np.tile(np.linspace(0,1,oversample_size), (self.crop_size,1)).T
    # Bottom feather
    if bottom < field.shape[0]:
      mask[self.crop_size-oversample_size:self.crop_size, :] = np.tile(np.linspace(1,0,oversample_size), (self.crop_size,1)).T
    # Left feather
    if left > 0:
      mask[:, 0:oversample_size] = np.tile(np.linspace(0,1,oversample_size), (self.crop_size, 1))
    # Right feather
    if right < field.shape[1]:
      mask[:, self.crop_size-oversample_size:self.crop_size] = np.tile(np.linspace(1,0,oversample_size), (self.crop_size, 1))

    # Top-left
    if top > 0 and left > 0:
      mask[0:oversample_size, 0:oversample_size] = np.tile(np.linspace(0,1,oversample_size), (oversample_size,1)).T*np.tile(np.linspace(0,1,oversample_size), (oversample_size, 1))
    # Top-right
    if top > 0 and right < field.shape[1]:
      mask[0:oversample_size, self.crop_size-oversample_size:self.crop_size] = np.fliplr(np.tile(np.linspace(0,1,oversample_size), (oversample_size,1)).T*np.tile(np.linspace(0,1,oversample_size), (oversample_size, 1)))
    # Bottom-left
    if bottom < field.shape[0] and left > 0:
      mask[self.crop_size-oversample_size:self.crop_size, 0:oversample_size] = np.fliplr(np.tile(np.linspace(1,0,oversample_size), (oversample_size,1)).T*np.tile(np.linspace(1,0,oversample_size), (oversample_size, 1)))
    # Bottom-right
    if bottom < field.shape[0] and right < field.shape[1]:
      mask[self.crop_size-oversample_size:self.crop_size, self.crop_size-oversample_size:self.crop_size] = np.tile(np.linspace(1,0,oversample_size), (oversample_size,1)).T*np.tile(np.linspace(1,0,oversample_size), (oversample_size, 1))

    for c_idx in range(crop.shape[2]):
      crop[...,c_idx] *= mask

    field[top:bottom, left:right] += crop[0:bottom-top, 0:right-left]

    return field

  def _get_model_metric(self, name):
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
      if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

      y_pred_f = tf.convert_to_tensor(y_pred[...,0:3])
      y_true_f = tf.cast(y_true[...,0:3], y_pred_f.dtype)

      # Get the cross_entropy for each entry
      ce = K.binary_crossentropy(y_true_f, y_pred_f, from_logits=from_logits)

      # If logits are provided then convert the predictions into probabilities
      if from_logits:
        pred_prob = tf.sigmoid(y_pred_f)
      else:
        pred_prob = y_pred_f

      p_t = (y_true_f * pred_prob) + ((1 - y_true_f) * (1 - pred_prob))
      alpha_factor = 1.0
      modulating_factor = 1.0

      if alpha:
        alpha = tf.cast(alpha, dtype=y_true_f.dtype)
        alpha_factor = y_true_f * alpha + (1 - y_true_f) * (1 - alpha)

      if gamma:
        gamma = tf.cast(gamma, dtype=y_true_f.dtype)
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
      y_pred_f = K.flatten(tf.cast(y_pred[...,1:3], dtype=tf.float32))
      y_true_f = K.flatten(tf.cast(y_true[...,1:3], y_pred_f.dtype))

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
      y_pred_f = K.flatten(tf.cast(y_pred[...,2], dtype=tf.float32))
      y_true_f = K.flatten(tf.cast(y_true[...,2], y_pred_f.dtype))
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
      y_pred_f = K.flatten(tf.cast(y_pred[...,1:3], dtype=tf.float32))
      y_true_f = K.flatten(tf.cast(y_true[...,1:3], y_pred_f.dtype))
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

    if name is None:
      return metrics

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
    metrics = self._get_model_metric(None)
    metrics['K'] = tf.keras.backend
    return metrics

  def _get_path(self):
    """
    Get the root path of this model

    Returns
    --------
    Path
    """
    return MNModel.models_root / type(self).__name__

  def train(self, train_path=None, val_path=None, batch_size=None, epochs=100, checkpoint_path=None, num_per_image=180, save_weights=True, save_path=None, load_weights=None):
    """
    Train a new model from scratch

    Parameters
    --------
    train_path : Path|str|None
      Path to training data root. If None, will use this package's training data.
    val_path : Path|str
      Path to validation data root. If None, will use this package's training data.
    batch_size : int|None
      Training batch size. If None, will default to self.batch_size
      (the prediction batch size)
    epochs : int
      The number of training epochs
    checkpoint_path : Path|str|None
      Where to save checkpoints during training, if not None
    num_per_image : int|None
      The number of crops to return per image. If None, will default to
      [img_width]//crop_size * [[img_height]]//crop_size. Because crops
      are randomly positioned and can be randomly augmented, more crops
      can be extracted from a given image than otherwise.
    save_weights : bool
      Whether to save the model weights
    save_path : str|Path|None
      Where to save model weights. If None, will default to models/[model_name]
    load_weights : str|Path|None
      If weights should be loaded prior to training, weights at the path specified by load_weights will be used

    Returns
    --------
    tf.Model
      The trained model
    tf.History
      Model training history
    """
    if batch_size is None:
      batch_size = self.batch_size

    if train_path is None:
      train_path = self.training_root

    if val_path is None:
      val_path = self.validation_root

    trainer = self._get_trainer(train_path, batch_size, num_per_image)
    validator = self._get_trainer(val_path, batch_size, num_per_image, augment=False)

    if checkpoint_path is not None:
      checkpoint_path = Path(checkpoint_path) / (type(self).__name__ + "-" + datetime.today().strftime('%Y-%m-%d') + "-{epoch:04d}.ckpt")

    metrics = self._get_custom_metrics()
    metric_funs = [ fun for k,fun in metrics.items() if k != "K" ]
    metric_funs.append("accuracy")

    model = self._build_model()
    if load_weights is not None:
      model.load_weights(load_weights)
    model.compile(
      optimizer=self._get_optimizer(),
      loss=self._get_loss_function(),
      metrics=metric_funs,
      loss_weights=self._get_loss_weights()
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='val_mean_iou',
      min_delta=1e-6,
      patience=10,
      verbose=1,
      mode='max',
      baseline=None,
      restore_best_weights=True
    )

    callbacks = [ reduce_lr, early_stop ]

    if checkpoint_path is not None:
      cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_weights_only=True,
        verbose=1,
        save_freq=20*batch_size
      )
      callbacks.append(cp_callback)

      model.save_weights(str(checkpoint_path).format(epoch=0))

    model_history = model.fit(
      trainer,
      epochs=epochs,
      steps_per_epoch=len(trainer),
      validation_data=validator,
      callbacks=callbacks
    )

    if save_weights:
      if save_path is None:
        save_path = self.models_root / type(self).__name__ / "final.weights.h5"
      elif not re.match(r"weights\.h5$", Path(save_path).name):
        raise Exception("`save_path` must end in weights.h5")

      model.save_weights(str(save_path))

    return model, model_history

  def _build_model(self):
    """
    Build the model
    
    Returns
    --------
    tf.keras.models.Model
    """
    raise MethodNotImplemented("I don't know how to build a model")

  def _get_trainer(self, data_path, batch_size, num_per_image, augment=True):
    """
    Return a trainer

    Parameters
    --------
    data_path : Path|str|None
      The path to the data sets
    batch_size : int
      Training batch size
    num_per_image : int
      The number of crops to return per training image
    augment : bool
      Whether to use image augmentation

    Returns
    --------
    tf.keras.utils.Sequence
    """
    return TFData(self.crop_size, data_path, batch_size, num_per_image, augment=augment)

  def _get_optimizer(self, lr=5e-4):
    """
    Return the keras optimizer to use for training

    Parameters
    --------
    lr : float
      Initial learning rate

    Returns
    --------
    tf.keras.optimizers.Optimizer
    """
    return tf.keras.optimizers.Adam(lr)

  def _get_loss_function(self):
    """
    Return the keras loss function to use

    Returns
    --------
    fun
    """
    return self._get_model_metric('sigmoid_focal_crossentropy')

  def _get_loss_weights(self):
    """
    Return the loss weights to use

    Returns
    --------
    list|None
    """
    return None

class MNClassifier(MNModel):
  """
  Base class for a MN pixel classifier.

  Attributes
  ----------
  models_root : Path
    Where model files are stored
  training_root : Path
    Where training data is stored
  testing_root : Path
    Where testing data is stored
  crop_size : int
    The input width and height of the model
  oversample_ratio : float
    The amount of overlap between crops when scanning across an image, as a proportion of crop_size
  batch_size : int
    Batch size for running predictions
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
  train(train_root=Path|str, val_root=Path|str, batch_size=None|int, epochs=100, checkpoint_path=Path|str|None, num_per_image=int|None)
    Build and train a model from scratch
  """

  class_type = "classifier"

  @staticmethod
  def get_available_models():
    """
    Return the list of available model classes

    Static method

    Returns
    --------
    list
    """
    if 'mnfinder.classifiers' in sys.modules:
      classifiers = sys.modules['mnfinder.classifiers']
    else:
      classifiers = importlib.import_module('mnfinder.classifiers')
    available_models = [ x[0] for x in inspect.getmembers(classifiers, inspect.isclass) if hasattr(x[1], 'class_type') and x[1].class_type == "classifier" and x[0] != "MNClassifier"]
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
    return model_name in MNClassifier.get_available_models()

  @staticmethod
  def get_model(model_name='Combined', weights_path=None, trained_model=None):
    """
    Returns an instance of the given model

    Static method

    Parameters
    --------
    model_name : str
      The model name. Defaults to the Combined class
    weights_path : Path|str|None
      Where to load the weights. If None, will load pretrained weights
    trained_model : tf.keras.Model|None
      To substitute an existing neural net model, specify it here
    
    Returns
    --------
    MNClassifier
    """
    available_models = MNClassifier.get_available_models()
    if model_name not in available_models:
      raise ModelNotFound("No such MN classifier: {}".format(model_name))
    try:
      if 'mnfinder.classifiers' in sys.modules:
        classifiers = sys.modules['mnfinder.classifiers']
      else:
        classifiers = importlib.import_module('mnfinder.classifiers')
      model = getattr(classifiers, model_name)
      return model(weights_path=weights_path, trained_model=trained_model)
    except:
      raise ModelNotLoaded("Could not load model: {}".format(model_name))

  def __init__(self, weights_path=None, trained_model=None, segmenter_name='DistSegmenter'):
    """
    Constructor

    Parameters
    --------
    weights_path : str|Path|None
      Where the model weights are stored. If None, defaults to models/[model_name]
    trained_model : tf.keras.Model|None
      If we wish to supply your own trained model, otherwise it will be loaded
    segmenter_name : str|None
      If a segmentation model should be loaded as well
    """
    super().__init__(weights_path=weights_path, trained_model=trained_model)

    if segmenter_name is not None and segmenter_name != type(self).__name__:
      self.segmenter = MNSegmenter.get_model(segmenter_name)
    else:
      self.segmenter = None

  def predict(self, img, skip_opening=None, expand_masks=None, use_argmax=None, area_thresh=250, return_raw_output=False, **kwargs):
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
    return_raw_output : bool
      Whether to return raw classifier and segmenter outputs
    
    Returns
    --------
    np.array
      The nucleus labels
    """
    if skip_opening is None:
      skip_opening = self.defaults.skip_opening

    if expand_masks is None:
      expand_masks = self.defaults.expand_masks

    if use_argmax is None:
      use_argmax = self.defaults.use_argmax

    img = self.normalize_dimensions(img)
    if img.shape[0] < self.crop_size or img.shape[1] < self.crop_size:
      raise ValueError("Image is smaller than minimum size of {}x{}".format(self.crop_size, self.crop_size))

    nucleus_pixels, mn_pixels, classifier_output = self._run_pixel_classifier(img, skip_opening=skip_opening, expand_masks=expand_masks, use_argmax=use_argmax, area_thresh=area_thresh)

    labels, segmenter_output = self._run_segmenter(img, nucleus_pixels, mn_pixels)

    if return_raw_output:
      return labels, classifier_output, segmenter_output
    else:
      return labels

  def _run_pixel_classifier(self, img, skip_opening=None, expand_masks=None, use_argmax=None, area_thresh=250):
    """
    Run classification for a given image

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
      The pixels classified as nuclei
    np.array
      The pixels classified as MN
    np.array
      The raw output of the field
    """
    field_output = self._get_field_predictions(img)

    field_classes = np.argmax(field_output[...,0:3], axis=-1).astype(np.uint8)
    nucleus_pixels = (field_classes == 1).astype(np.uint8)
    mn_pixels = np.zeros_like(nucleus_pixels)
    if use_argmax:
      mn_pixels[(field_classes == 2)] = 1
    else:
      mn_pixels[((field_output[...,0] < self.bg_max) & (field_output[...,2] > self.fg_min))] = 1

    inner_nucleus_labels = clear_border(nucleus_pixels)
    inner_nucleus_labels = label(inner_nucleus_labels)

    if area_thresh is not False and area_thresh > 0:
      possible_mn_info = pd.DataFrame(regionprops_table(inner_nucleus_labels, properties=('label', 'area')))
      switch_labels = possible_mn_info['label'].loc[(possible_mn_info['area'] < area_thresh)]
      mn_pixels[np.isin(inner_nucleus_labels, switch_labels)] = 1
      nucleus_pixels[np.isin(inner_nucleus_labels, switch_labels)] = 0

    if not skip_opening:
      mn_pixels = binary_opening(mn_pixels, footprint=disk(self.defaults.opening_radius)).astype(np.uint8)
    mn_pixels = clear_border(mn_pixels)

    if expand_masks:
      mn_pixels = self._expand_masks(mn_pixels)

    nucleus_pixels[mn_pixels > 0] = 0

    return nucleus_pixels, mn_pixels, field_output

  def _run_segmenter(self, img, nucleus_pixels, mn_pixels):
    """
    Run classification for a given image

    Parameters
    --------
    img : np.array
      The image to predict
    nucleus_pixels : np.array
      Pixels classed as nuclei
    mn_pixels : np.array
      Pixels classed as MN
    
    Returns
    --------
    np.array
      An array with 3 channels: nucleus labels, MN labelled with their
      assigned nucleus label, unique MN labels
    np.array
      The raw output of the segmenter
    """
    if self.segmenter is None:
      nucleus_labels = label(nucleus_pixels, connectivity=1)
      mn_labels = label(mn_pixels, connectivity=1)
      mn_nuc_labels = nucleus_labels.copy()
      
      nuc_info = regionprops_table(nucleus_labels, properties=('label', 'centroid'))
      mn_info = regionprops_table(mn_labels, properties=('label', 'centroid'))

      # Find nearest nuc and relabel
      nuc_tree = spatial.KDTree(list(zip(nuc_info['centroid-1'], nuc_info['centroid-0'])))
      for i,x in enumerate(mn_info['centroid-1']):
        y = mn_info['centroid-0'][i]
        res = nuc_tree.query([ x, y ], k=1)
        mn_nuc_labels[mn_labels == mn_info['label'][i]] = nuc_info['label'][res[1]]

      nucleus_labels = clear_border(nucleus_labels)
      mn_labels = clear_border(mn_labels)
      mn_nuc_labels = clear_border(mn_nuc_labels)
    else:
      labels, segmenter_output= self.segmenter.segment(img, nucleus_pixels, mn_pixels)

    return labels, segmenter_output

  def _expand_masks(self, mn_pixels):
    """
    Returns the convex hulls of mn_labels

    Parameters
    --------
    mn_pixels : np.array
      Img with mn-classed pixels == 1
      
    Returns
    --------
    np.array
      The modified labels
    """
    mn_labels = label(mn_pixels, connectivity=1)
    if len(np.unique(mn_labels)) < 2:
      return mn_pixels

    new_mn_pixels = np.zeros_like(mn_pixels)
    for mn_label in np.unique(mn_labels):
      if mn_label == 0:
        continue
      img_copy = np.zeros_like(mn_labels, dtype=bool)
      img_copy[mn_labels == mn_label] = True
      img_copy = convex_hull_image(img_copy)
      new_mn_pixels[img_copy] = 1

    return new_mn_pixels

class MNSegmenter(MNModel):
  """
  Base class for a MN pixel classifier.

  Attributes
  ----------
  models_root : Path
    Where model files are stored
  training_root : Path
    Where training data is stored
  testing_root : Path
    Where testing data is stored
  crop_size : int
    The input width and height of the model
  oversample_ratio : float
    The amount of overlap between crops when scanning across an image, as a proportion of crop_size
  batch_size : int
    Batch size for running predictions
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
  segment(img=np.array, nucleus_pixels=np.array, mn_pixels=np.array)
    Segments nuclei and assigns MN to nuclei
  train(train_root=Path|str, val_root=Path|str, batch_size=None|int, epochs=100, checkpoint_path=Path|str|None, num_per_image=int|None)
    Build and train a model from scratch
  """

  class_type = "segmenter"

  @staticmethod
  def get_available_models():
    """
    Return the list of available model classes

    Static method

    Returns
    --------
    list
    """
    if 'mnfinder.segmenters' in sys.modules:
      segmenters = sys.modules['mnfinder.segmenters']
    else:
      segmenters = importlib.import_module('mnfinder.segmenters')
    available_models = [ x[0] for x in inspect.getmembers(segmenters, inspect.isclass) if hasattr(x[1], 'class_type') and x[1].class_type == "segmenter" and x[0] != "MNSegmenter" ]
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
    return model_name in MNSegmenter.get_available_models()

  @staticmethod
  def get_model(model_name='DistSegmenter', weights_path=None, trained_model=None):
    """
    Returns an instance of the given model

    Static method

    Parameters
    --------
    model_name : str
      The model name. Defaults to the Attention class
    weights_path : Path|str|None
      Where to load the weights. If None, will load pretrained weights
    trained_model : tf.keras.Model|None
      To substitute an existing neural net model, specify it here
    
    Returns
    --------
    MNSegmenter
    """
    available_models = MNSegmenter.get_available_models()
    if model_name not in available_models:
      raise ModelNotFound("No such MN segmenter: {}".format(model_name))
    try:
      if 'mnfinder.segmenters' in sys.modules:
        segmenters = sys.modules['mnfinder.segmenters']
      else:
        segmenters = importlib.import_module('mnfinder.segmenters')

      model = getattr(segmenters, model_name)

      return model(weights_path=weights_path, trained_model=trained_model)
    except:
      raise ModelNotLoaded("Could not load model: {}".format(model_name))

  def segment(self, img, nucleus_pixels, mn_pixels):
    """
    Labels individual nuclei and MN, and associates MN to nuclei

    Parameters
    --------
    img : np.array
      The image to predict
    nucleus_pixels : np.array
      The pixels classified as nuclei by an MNClassifier
    mn_pixels : np.array
      The pixels classified as MN by an MNClassifier
    
    Returns
    --------
    np.array
      An array with 3 channels: nucleus labels, MN labelled with their
      assigned nucleus label, unique MN labels
    np.array
      Raw output
    """
    field_output = self._get_field_predictions(img)

    # Watershed
    distances = field_output[...,1]
    bin_distances = np.zeros_like(distances).astype(np.uint8)
    bin_distances[distances > threshold_li(distances)] = 1
    bin_distances = binary_dilation(bin_distances, disk(2))

    edges = field_output[...,2]*binary_dilation(bin_distances, disk(1))

    combined = edges-distances
    centroids = peak_local_max(-combined, footprint=disk(10), labels=bin_distances)

    markers = np.zeros(distances.shape, dtype=bool)
    markers[tuple(centroids.T)] = True
    markers = label(markers)

    labels = watershed(combined, markers, mask=bin_distances)

    # Merge labels that are bounded with no predicted edges
    edge_skeleton = np.zeros(( field_output.shape[0], field_output.shape[1] ), dtype=bool)
    edge_skeleton[edges > threshold_yen(edges)] = 1
    edge_skeleton = binary_dilation(skeletonize(edge_skeleton), disk(1))

    merging = True
    merge_count = 0
    while merging:
      if merge_count > 1000:
        break

      merge_count += 1

      boundaries = find_boundaries(labels, connectivity=2, mode='outer')
      boundaries[(labels == 0)] = 0
      boundaries[(edge_skeleton > 0)] = 0
      boundaries = label(boundaries)

      merging = False
      for b_label in np.unique(boundaries):
        if b_label == 0:
          continue

        to_merge, counts = np.unique(labels[boundaries == b_label], return_counts=True)
        to_merge = to_merge[counts > 10]
        if len(to_merge) < 2:
          continue

        labels[np.isin(labels, to_merge)] = to_merge[0]
        merging = True
        break

    nucleus_labels = labels.copy()
    nucleus_labels[nucleus_pixels == 0] = 0

    mn_labels = label(mn_pixels, connectivity=1)

    mn_nuc_labels = np.zeros( (labels.shape[0], labels.shape[1], 3), dtype=np.uint16)
    mn_nuc_labels[...,0] = nucleus_labels.copy()
    
    mn_nuc_labels[...,1] = labels.copy()
    mn_nuc_labels[mn_pixels == 0, 1] = 0

    # Merge in any MN pixels not covered by labels
    nuc_info = regionprops_table(nucleus_labels, properties=('label', 'centroid'))
    nuc_tree = spatial.KDTree(list(zip(nuc_info['centroid-1'], nuc_info['centroid-0'])))

    orphan_mn_labels = np.unique(mn_labels[(mn_nuc_labels[...,1] == 0) & (mn_labels != 0)])
    if len(orphan_mn_labels) > 0:
      mn_info = pd.DataFrame(regionprops_table(mn_labels, properties=('label', 'centroid')))
      for orphan_label in orphan_mn_labels:
        new_labels, counts = np.unique(mn_nuc_labels[(mn_labels == orphan_label) & (mn_nuc_labels[...,1] != 0),1], return_counts=True)

        if len(new_labels) > 0:
          # Assign all pixels belonging to orphan label to the mn_nuc_label with the most overlap
          mn_nuc_labels[mn_labels == orphan_label, 1] = new_labels[np.argmax(counts)]

        else:
          # If there is no overlap, find the closest nucleus
          x = mn_info.loc[mn_info.label == orphan_label, 'centroid-1'].iloc[0]
          y = mn_info.loc[mn_info.label == orphan_label, 'centroid-0'].iloc[0]
          res = nuc_tree.query([ x, y ], k=1)
          mn_nuc_labels[mn_labels == orphan_label,1] = nuc_info['label'][res[1]]

    # Reassign any MN that were segmented, but have no nucleus
    orphan_mn_nuc_labels = np.setdiff1d(np.unique(mn_nuc_labels[...,1]), np.unique(mn_nuc_labels[...,0]))
    if len(orphan_mn_nuc_labels) > 0:
      mn_info = pd.DataFrame(regionprops_table(mn_nuc_labels[...,1], properties=('label', 'centroid')))
      for orphan_label in orphan_mn_nuc_labels:
        x = mn_info.loc[mn_info.label == orphan_label, 'centroid-1'].iloc[0]
        y = mn_info.loc[mn_info.label == orphan_label, 'centroid-0'].iloc[0]
        res = nuc_tree.query([ x, y ], k=1)
        mn_nuc_labels[mn_nuc_labels[...,1] == orphan_label,1] = nuc_info['label'][res[1]]

    mn_nuc_labels[...,0] = clear_border(mn_nuc_labels[...,0])
    mn_nuc_labels[...,1] = clear_border(mn_nuc_labels[...,1])
    mn_nuc_labels[...,2] = clear_border(mn_labels)

    mn_nuc_labels[...,0] = expand_labels(mn_nuc_labels[...,0], 1)
    mn_nuc_labels[...,0][mn_nuc_labels[...,1] > 0] = 0

    return mn_nuc_labels, field_output


class TrainingDataGenerator:
  """
  Generates training data

  This assumes the following directory structure:
    [data_path]/
      [dataset1]/
        mn_masks/
        nucleus_masks/
        images/
      [dataset2]/
        mn_masks/
        nucleus_masks/
        images/
      ...

  MN masks, nucleus masks, and the associated image must share the
  same name, aside from the suffix. Images and masks can be any format 
  readable by PIL.Image.open() or TiffFile.imread()
  
  This class functions as an iterator, and will iterate through all
  training data in random order, generating randomly positioned crops 
  until all images have been cycled through.

  Images may be augmented, MN may have a border class drawn around them,
  and images without any nuclei or MN masks can be skipped during training.

  Attributes
  ----------
  crop_size : int
    Width and height of crops
  data_path : Path
    The root path to training data
  augment : bool
    Whether to perform image augmentation
  draw_border : bool
    Whether to draw an MN border class around MN segments
  skip_empty : bool
    Whether to only return crops that have nucleus or MN segments
  num_per_image : int|None
    The number of crops to return per image. If None, will default to
    [img_width]//crop_size * [[img_height]]//crop_size. Because crops
    are randomly positioned and can be randomly augmented, more crops
    can be extracted from a given image than otherwise.

  Static methods
  --------
  open_mask(path=Path|str)
    Gets a mask
  open_image(path=Path|str)
    Returns an image as a list of individual channels + their sobel filters
  get_combined_mask(mn_mask_path=Path|str, pn_mask_path=Path|str)
    Returns a single numpy array with nuclei = 1 and MN = 2
  

  Public methods
  --------
  crop_image(img_idx=int)
    Generates crops of both images and masks
  """
  def __init__(self, crop_size, data_path, augment=False, draw_border=False, skip_empty=True, num_per_image=None, use_dist_masks=False, post_hooks=None):
    """
    Constructor
    
    Parameters
    --------
    crop_size : int
      Crop width and height
    data_path : Path|str
      Path to root of training data
    augment : bool
      Whether to perform image augmentation
    draw_border : bool
      Whether to inject an additional MN border class, drawn around MN segments
    skip_empty : bool
      Whether to only return training data that has nucleus or MN segments
    num_per_image : int|None
      The number of crops to return per image. If None, will default to
      [img_width]//crop_size * [[img_height]]//crop_size. Because crops
      are randomly positioned and can be randomly augmented, more crops
      can be extracted from a given image than otherwise.
    post_hooks : None|list
      A list of post-processing functions to perform on images
    """
    data_path = Path(data_path).resolve()
    if not data_path.exists():
      raise FileNotFoundError("Path `{}` does not exist".format(str(data_path)))
    self.crop_size = crop_size
    self.data_path = data_path
    self.augment = augment
    self.draw_border = draw_border
    self.skip_empty = skip_empty
    self.num_per_image = num_per_image
    self.post_hooks = post_hooks
    self.use_dist_masks = use_dist_masks

    dirs = [ x for x in data_path.iterdir() if x.is_dir() ]
    print("Located {} directories...".format(len(dirs)))

    self.image_paths = []
    self.mn_masks_paths = []
    self.pn_masks_paths = []

    self.distance_masks = {}

    mn_mask_dir_name = "mn_masks"
    pn_mask_dir_name = "nucleus_masks"
    image_dir_name = "images"

    for d in tqdm(dirs):
      mask_dir = d / mn_mask_dir_name
      pn_dir = d / pn_mask_dir_name
      image_dir = d / image_dir_name

      mask_list = [ x for x in mask_dir.iterdir() if x.is_file() and x.name[0] != "." ]

      for x in mask_list:
        self.mn_masks_paths.append(mask_dir / x.name)
        self.pn_masks_paths.append(pn_dir / x.name)
        self.image_paths.append(image_dir / (x.stem + ".tif"))
        if use_dist_masks:
          self.distance_masks[str(mask_dir / x.name)] = self._get_distance_masks(
            pn_dir / x.name,
            mask_dir / x.name
          )

  def __iter__(self):
    """
    Iterator

    Allows this class to be called as an iterator to return 
    training data
    
    Returns
    --------
    dict
      Dictionary with keys:
        image : np.array
          The crop
        segmentation_mask : np.array
          Ground truth
        source : Path
          The image source
        coords : list
          Where this crop came from in the image
    """
    possible_imgs = list(range(len(self.image_paths)))
    random.shuffle(possible_imgs)
    for img_idx in possible_imgs:
      data_points = self.crop_image(img_idx)
      if self.post_hooks is not None:
        for fun in self.post_hooks:
          data_points = fun(data_points)

      for data_point in data_points:
        yield data_point

  @staticmethod
  def get_full_mask(pn_mask, mn_mask):
    full_mask = np.zeros((pn_mask.shape[0], pn_mask.shape[1], 3), dtype=np.uint16)
    pn_colors = np.unique(np.reshape(pn_mask, (pn_mask.shape[0]*pn_mask.shape[1],4)), axis=0)
    mn_colors = np.unique(np.reshape(mn_mask, (mn_mask.shape[0]*mn_mask.shape[1],4)), axis=0)
    mn_id = 1
    for color_id in pn_colors[...,2]:
      if color_id == 0:
        continue
      full_mask[pn_mask[...,2] == color_id] = [ 1, color_id, 0 ]

      for mn_color in mn_colors[mn_colors[...,2] == color_id][...,1]:
        full_mask[(mn_mask[...,2] == color_id) & (mn_mask[...,1] == mn_color)] = [ 2, color_id, mn_id ]
        mn_id += 1

    full_mask[...,2] = label(full_mask[...,2])

    return full_mask

  def _get_distance_masks(self, pn_mask_path, mn_mask_path):
    pn_mask = np.array(Image.open(pn_mask_path))
    mn_mask = np.array(Image.open(mn_mask_path))

    full_mask = TrainingDataGenerator.get_full_mask(pn_mask, mn_mask)

    hulls = np.zeros((full_mask.shape[0], full_mask.shape[1])).astype(np.uint16)
    distances = np.zeros((full_mask.shape[0], full_mask.shape[1], 2)).astype(np.float64)

    for cell_id in np.unique(full_mask[...,1]):
      if cell_id == 0:
        continue
      y, x = np.where(full_mask[...,1] == cell_id)
      coords = np.array(list(zip(x,y)))
      ch = ConcaveHull()
      ch.loadpoints(coords.tolist())
      ch.calculatehull(tol=100)
      vertices = np.expand_dims(ch.boundary_points(), axis=1).astype(np.int32)
      hulls = cv2.fillPoly(hulls, [vertices], int(cell_id))

    for hull_id in np.unique(hulls):
      if hull_id == 0:
        continue
      tmp = np.zeros(( distances.shape[0], distances.shape[1] )).astype(np.float64)
      tmp[hulls == hull_id] = 1
      distances[...,0] += rescale_intensity(distance_transform_edt(tmp), out_range=(0,1))

      tmp = np.ones(( distances.shape[0], distances.shape[1] )).astype(np.uint8)
      tmp[(hulls != hull_id) & (hulls != 0)] = 0
      tmp = distance_transform_edt(tmp)
      tmp[hulls != hull_id] = 0
      distances[...,1] += (1-rescale_intensity(tmp, out_range=(0,1)))**4
    
    distances[...,1] -= np.min(distances[...,1])
    distances[...,1][hulls == 0] = 0
    return distances

  def crop_image(self, img_idx):
    """
    Generate crops from a given image
    
    Parameters
    --------
    img_idx : int
      The index of which image to crop

    Returns
    --------
    list
      List of dicts as described in __iter__()
    """
    image_path = self.image_paths[img_idx]
    mn_mask_path = self.mn_masks_paths[img_idx]
    pn_mask_path = self.pn_masks_paths[img_idx]

    channels = TrainingDataGenerator.open_image(image_path)
    if self.use_dist_masks:
      mask = self.get_dual_mask(mn_mask_path, pn_mask_path)
    else:
      mask = self.get_combined_mask(mn_mask_path, pn_mask_path)
      if self.draw_border:
        mn_mask = mask[...,2].copy()
        mn_mask[mn_mask[...,0] != 2] = 0
        outside = dilation(mn_mask, disk(2))
        mask = np.expand_dims(mask, axis=-1)
        mask[(outside != 0) & (mask != 2),3] = 1 # Generate boundaries

    datapoints = self._crop_image_random(channels, mask)

    update = { 'source': image_path }
    datapoints = [ {**x, **update} for x in datapoints ]

    return datapoints

  def _crop_image_random(self, channels, mask):
    """
    Generate crops from a given image
    
    Parameters
    --------
    channels : list
      The individual channels of an image, + its sobel filters
    mask : np.array
      The combined ground truth with nuclei = 1, MN = 2, and (optionally)
      MN borders = 3

    Returns
    --------
    list
      List of dicts
    """
    width = channels[0].shape[1]
    height = channels[0].shape[0]

    datapoints = []
    if self.num_per_image is None:
      num_per_image = (width//crop_size)*(height//crop_size)
    else:
      num_per_image = self.num_per_image

    while len(datapoints) < num_per_image:
      this_x = random.randrange(width)
      this_y = random.randrange(height)

      left = this_x
      right = left + self.crop_size
      top = this_y
      bottom = top + self.crop_size

      if right > width:
        right = width
      if bottom > height:
        bottom = height

      crop_height = bottom-top
      crop_width = right-left

      crop = np.zeros(( self.crop_size, self.crop_size, len(channels) ), dtype=channels[0].dtype)
      if len(mask.shape) == 3:
        crop_mask = np.zeros(( self.crop_size, self.crop_size, mask.shape[2] ), dtype=mask.dtype)
      else:
        crop_mask = np.zeros(( self.crop_size, self.crop_size ), dtype=mask.dtype)

      for i,channel in enumerate(channels):
        crop[0:crop_height,0:crop_width,i] = channel[ top:bottom, left:right ]
      crop_mask[0:crop_height, 0:crop_width] = mask[ top:bottom, left:right ]

      datapoint = {
        'image': crop,
        'segmentation_mask': crop_mask,
        'coords': (left, right, top, bottom)
      }

      if self.augment:
        datapoint = self._augment_datapoint(datapoint)

      if self.use_dist_masks and self.skip_empty and np.all(datapoint['segmentation_mask'] == 0):
        continue
      elif self.skip_empty and np.sum(datapoint['segmentation_mask']) <= 0:
        continue

      datapoints.append(datapoint)

    return datapoints

  def _augment_datapoint(self, datapoint):
    """
    Augment a given crop
    
    Parameters
    --------
    datapoint : dict
      Dict containing the image and segmentation mask
    
    Returns
    --------
    dict
      The modified datapoint
    """
    aug = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.Rotate(p=0.5, limit=(-90,270), border_mode=cv2.BORDER_REFLECT),
      A.Transpose(p=0.5),
      # A.MaskDropout(p=0.5, max_objects=(0,5), image_fill_value=np.median(datapoint['image'][...,0]))
      # A.Perspective(p=0.3, scale=[0.05, 0.08]),
      # A.ElasticTransform(p=1.0, alpha=12, sigma=15, alpha_affine=5, border_mode=cv2.BORDER_REFLECT, value=0)
    ])

    augmented = aug(image=datapoint['image'], mask=datapoint['segmentation_mask'])
    aug_image = augmented['image'][0:self.crop_size,0:self.crop_size,:]
    
    return {
      'image': aug_image,
      'segmentation_mask': augmented['mask'].astype(datapoint['segmentation_mask'].dtype),
      'coords': datapoint['coords']
    }

  @staticmethod
  def open_mask(path):
    """
    Get nucleus or MN mask
    
    Parameters
    --------
    path : Path|str
      Path to the mask
    
    Returns
    --------
    np.array
      The mask
    """
    if path.suffix.lower() == "tiff" or path.suffix.lower() == "tif":
      img = tifffile.imread(path)
    else:
      img = np.array(Image.open(path))

    img = MNModel.normalize_dimensions(img)
    return img

  @staticmethod
  def open_image(path):
    """
    Get image
    
    Parameters
    --------
    path : Path|str
      Path to the image
    
    Returns
    --------
    list
      The individual channels split into a list +
      sobel filters on each channel
    """
    if path.suffix.lower() == "tiff" or path.suffix.lower() == "tif":
      img = tifffile.imread(path)
    else:
      img = np.array(Image.open(path))

    img = MNModel.normalize_dimensions(img)
    channels = []
    edges = []
    for channel in range(img.shape[2]):
      channels.append(MNModel.normalize_image(img[...,channel]))

    edges = [ sobel(x) for x in channels ]
    edges = [ MNModel.normalize_image(x) for x in edges ]

    channels += edges
    return channels

  def get_dual_mask(self, mn_mask_path, pn_mask_path):
    """
    Read nucleis and MN masks and combine into a single image

    Nuclei = 1, MN = 2
    
    Parameters
    --------
    mn_mask_path : Path|str
      Path to the MN mask
    pn_mask_path : Path|str
      Path to the nucleus mask
    
    Returns
    --------
    np.array
    """
    mask = self.get_combined_mask(mn_mask_path, pn_mask_path)

    return np.stack([ mask[...,0], mask[...,1], mask[...,2], self.distance_masks[str(mn_mask_path)][...,0], self.distance_masks[str(mn_mask_path)][...,1]], axis=-1)

  def get_combined_mask(self, mn_mask_path, pn_mask_path):
    """
    Read nucleis and MN masks and combine into a single image

    Nuclei = 1, MN = 2
    
    Parameters
    --------
    mn_mask_path : Path|str
      Path to the MN mask
    pn_mask_path : Path|str
      Path to the nucleus mask
    
    Returns
    --------
    np.array
    """
    pn_details = TrainingDataGenerator.open_mask(pn_mask_path)
    mn_details = TrainingDataGenerator.open_mask(mn_mask_path)

    nuc_mask = np.zeros(( pn_details.shape[0], pn_details.shape[1] ), dtype=np.float64)
    mn_mask = np.zeros_like(nuc_mask)
    bg_mask = np.zeros_like(nuc_mask)

    nuc_mask[(pn_details[...,0] > 0)] = 1 # Nucleus
    mn_mask[(mn_details[...,0] > 0)] = 1 # MN
    bg_mask[(pn_details[...,0] == 0) & (mn_details[...,0] == 0)] = 1

    return np.stack([ bg_mask, nuc_mask, mn_mask ], axis=-1)

class TFData(Sequence):
  """
  Provides training and validation data during training
  """
  def __init__(self, crop_size, data_path, batch_size, num_per_image, use_dist_masks=False, workers=1, use_multiprocessing=False, max_queue_size=10, **kwargs):
    """
    Load a TrainingDataGenerator class

    Parameters
    --------
    crop_size : int
      Crop size
    data_path : Path|str
      Path to data sets
    batch_size : int
      Batch size for training
    num_per_image : int
      The number of crops to generate / image
    """
    super().__init__(workers=1, use_multiprocessing=False, max_queue_size=10)

    self.dg = TrainingDataGenerator(
      crop_size,
      data_path,
      num_per_image=num_per_image,
      use_dist_masks=use_dist_masks,
      **kwargs
    )
    self.num_images = len(self.dg.mn_masks_paths)
    self.num_per_image = num_per_image
    self.batch_size = batch_size

  def __len__(self):
    return self.num_images*self.num_per_image

  def __getitem__(self, idx):
    """
    Return a batch of training data

    Parameters
    --------
    idx : int
      The index of the batch to return
    
    Returns
    --------
    np.array, np.array
      The training data and ground truth as arrays
    """
    batch_x = []
    batch_y = []

    i = 0
    for dp in self.dg:
      batch_x.append(dp['image'])
      batch_y.append(dp['segmentation_mask'])
      i += 1
      if i >= self.batch_size:
        break

    return np.array(batch_x), np.array(batch_y)

class IncorrectDimensions(Exception):
  "Images must be (x,y,c) or (x,y)"
  pass

class ModelNotFound(Exception):
  "That model could not be found"
  pass

class ModelNotLoaded(Exception):
  "That model could not be loaded"
  pass

class MethodNotImplemented(Exception):
  "That method has not been implemented"
  pass