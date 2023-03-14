from stardist.models import StarDist2D
from csbdeep.utils import normalize
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as K
import json
from skimage.filters import sobel
from skimage.measure import regionprops_table, label, find_contours
from skimage.morphology import disk, binary_opening, opening, binary_erosion
from skimage.exposure import rescale_intensity
from skimage.color import label2rgb
import pandas as pd
import numpy as np
import cv2
from tifffile import TiffWriter, TiffFile
import requests
import tarfile
from tqdm import tqdm

import matplotlib.pyplot as plt

class MNModel:

  models_root = (Path(__file__) / "../models").resolve()

  @staticmethod
  def get_available_models():
    available_models = [ 'DiffSobel', 'FocalLoss', 'FocalLoss4Class', 'FocalLoss4Class2', 'FocalLoss4Class3', 'Combined' ]

    return available_models

  @staticmethod
  def get_model(model_name):
    available_models = MNModel.get_available_models()
    if model_name not in available_models:
      raise ModelNotFound("No such MN model")
    try:
      model = globals()[model_name]
      return model()
    except:
      raise ModelNotLoaded("Could not load model")

  @staticmethod
  def normalize_image(img, use_csbdeep=False):
    if use_csbdeep:
      return normalize(img, 0.0, 100.0)
    p2, p98 = np.percentile(img, (5, 99))
    img = rescale_intensity(img, in_range=(p2, p98), out_range=np.uint8).astype(np.float64)
    return img/255.0

  def __init__(self):
    self.nuc_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    self.crop_size = 96
    self.batch_size = 64
    self.name = self.__class__.__name__
    self._load_model()

  def _load_model(self):
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

  def predict_nuclei(self, img, p_min=2.0, p_max=99.8, prob_thresh=0.479071, nms_thresh=0.4, area_thresh=200, **kwargs):
    img = self.normalize_dimensions(img)
    nucleus_labels, details = self.nuc_model.predict_instances(
      normalize(img[...,0], p_min, p_max),
      prob_thresh=prob_thresh,
      nms_thresh=nms_thresh
    )

    nucleus_regions = pd.DataFrame(regionprops_table(nucleus_labels, properties=('label', 'area')))
    to_remove = nucleus_regions['label'].loc[nucleus_regions['area'] < area_thresh].tolist()
    nucleus_labels[np.isin(nucleus_labels, to_remove)] = 0

    return nucleus_labels

  def predict_mn(self, img, use_argmax=True, skip_opening=False, **kwargs):
    img = self.normalize_dimensions(img)
    if img.shape[0] < self.crop_size or img.shape[1] < self.crop_size:
      raise ValueError("Image is smaller than minimum size of {}x{}".format(self.crop_size, self.crop_size))

    coords, dataset, predictions = self._get_mn_predictions(img)
    num_channels = predictions[0].shape[2]
    field_output = np.zeros(( img.shape[0], img.shape[1], num_channels ), dtype=np.float64)
    for idx, batch in enumerate(dataset):
      left   = coords[idx][0]
      right  = coords[idx][1]
      top    = coords[idx][2]
      bottom = coords[idx][3]

      field_output[top:bottom, left:right] = predictions[idx]

    if not use_argmax:
      field_prediction = rescale_intensity(field_output, out_range=np.uint8)
      field_prediction = ((field_prediction[...,0] < self.bg_thresh) & (field_prediction[...,2] > self.fg_thresh))
    else:
      field_prediction = (np.argmax(field_output, axis=-1) == 2).astype(np.uint8)

    if not skip_opening:
      field_prediction = binary_opening(field_prediction, footprint=disk(1)).astype(np.uint8)
    field_labels = label(field_prediction, connectivity=1)

    return field_labels, field_output

  def predict_field(self, img, **kwargs):
    nuclei_labels = self.predict_nuclei(img, **kwargs)
    mn_labels, mn_raw = self.predict_mn(img, skip_opening=True, **kwargs)

    binary_nuclei = (nuclei_labels > 0)
    
    # Filter out MN within nucleus masks
    mn_labels[binary_nuclei] = 0
    mn_labels = opening(mn_labels, footprint=disk(1)).astype(np.uint8)

    return nuclei_labels, mn_labels, mn_raw

  @staticmethod
  def eval_mn_prediction(mn_true_masks, mn_labels):
    true_mn_labels = label(mn_true_masks, connectivity=1)

    intact_mn = np.zeros_like(true_mn_labels)
    ruptured_mn = np.zeros_like(true_mn_labels)
    intact_mn[(mn_true_masks == 1)] = 1
    ruptured_mn[(mn_true_masks == 2)] = 1

    mn_df = {
      'true_mn_label': [],
      'intact': [],
      'found': []
    }

    pred_df = {
      'pred_mn_label': [],
      'exists': []
    }

    summary_df = {
      'num_mn': [],
      'num_intact_mn': [],
      'num_ruptured_mn': [],
      'num_predictions': [],
      'num_mn_found': [],
      'num_intact_mn_found': [],
      'num_ruptured_mn_found': [],
      'iou': []
    }

    for mn_label in np.unique(true_mn_labels):
      if mn_label == 0:
        continue

      mn_df['true_mn_label'].append(mn_label)
      if np.sum(intact_mn[(true_mn_labels == mn_label)]) > 0:
        mn_df['intact'].append(True)
      else:
        mn_df['intact'].append(False)

      if np.sum(mn_labels[(true_mn_labels == mn_label)]) > 0:
        mn_df['found'].append(True)
      else:
        mn_df['found'].append(False)

    for mn_label in np.unique(mn_labels):
      pred_df['pred_mn_label'].append(mn_label)
      if np.sum(true_mn_labels[(mn_labels == mn_label)]) > 0:
        pred_df['exists'].append(True)
      else:
        pred_df['exists'].append(False)

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

    summary_df = pd.DataFrame(summary_df)

    summary_df['ppv'] = summary_df['num_mn_found']/summary_df['num_predictions']
    summary_df['recall'] = summary_df['num_mn_found']/summary_df['num_mn']
    summary_df['intact_recall'] = summary_df['num_intact_mn_found']/summary_df['num_intact_mn']
    summary_df['ruptured_recall'] = summary_df['num_ruptured_mn_found']/summary_df['num_ruptured_mn']

    return mn_df, pred_df, summary_df


  @staticmethod
  def _get_model_metric(name):
    def sigmoid_focal_crossentropy(
      y_true,
      y_pred,
      alpha = 0.25,
      gamma = 2.0,
      from_logits = False,
    ):
      """Implements the focal loss function.
      Focal loss was first introduced in the RetinaNet paper
      (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
      classification when you have highly imbalanced classes. It down-weights
      well-classified examples and focuses on hard examples. The loss value is
      much higher for a sample which is misclassified by the classifier as compared
      to the loss value corresponding to a well-classified example. One of the
      best use-cases of focal loss is its usage in object detection where the
      imbalance between the background class and other classes is extremely high.
      Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
      Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
      """
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
      return sigmoid_focal_crossentropy(y_true, y_pred)

    def tversky(y_true, y_pred, smooth=1, alpha=0.7):
      y_true_pos = tf.cast(K.flatten(K.one_hot(tf.cast(y_true, dtype=tf.uint8), num_classes=3)[...,1:]), dtype=tf.float32)
      y_pred_pos = K.flatten(tf.cast(y_pred[...,1:], dtype=tf.float32))
      true_pos = K.sum(y_true_pos * y_pred_pos)
      false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
      false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
      return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                    (1 - alpha) * false_pos + smooth)

    def dice_coef(y_true, y_pred, smooth=1):
      y_true_f = tf.cast(K.flatten(K.one_hot(tf.cast(y_true, dtype=tf.uint8), num_classes=3)[...,1:]), dtype=tf.float32)
      y_pred_f = K.flatten(tf.cast(y_pred[...,1:], dtype=tf.float32))
      intersection = K.sum(y_true_f * y_pred_f, axis=-1)
      denom = K.sum(y_true_f + y_pred_f, axis=-1)
      return K.mean((2. * intersection / (denom + smooth)))

    metrics = {
      'sigmoid_focal_crossentropy': sigmoid_focal_crossentropy,
      'sigmoid_focal_crossentropy_loss': sigmoid_focal_crossentropy_loss,
      'tversky': tversky,
      'dice_coef': dice_coef
    }

    return metrics[name]

  def normalize_dimensions(self, img):
    if len(img.shape) == 3:
      return img
    if len(img.shape) == 2:
      return np.stack([ img ], axis=-1)
    raise IncorrectDimensions()

  def _get_custom_metrics(self):
    return { 
      'sigmoid_focal_crossentropy_loss': self._get_model_metric('sigmoid_focal_crossentropy_loss'), 
      'sigmoid_focal_crossentropy': self._get_model_metric('sigmoid_focal_crossentropy') 
    }

  def _get_path(self):
    return MNModel.models_root / self.name

  def _get_mn_predictions(self, img):
    tensors = []
    coords = []
    num_channels = img.shape[2]
    crops = self._get_image_crops(img)

    sobel_idx = num_channels

    for crop in crops:
      tensors.append(tf.convert_to_tensor(
        np.stack([ crop['image'][...,0], crop['image'][...,sobel_idx] ], axis=-1)
      ))
      coords.append(crop['coords'])

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset_batchs = dataset.batch(self.batch_size)
    predictions = self.trained_model.predict(dataset_batchs)

    return coords, dataset, predictions

  def _get_image_crops(self, img):
    channels = []
    edges = []
    for channel in range(img.shape[2]):
      channels.append(self.normalize_image(img[...,channel]))

    edges = [ sobel(x) for x in channels ]
    edges = [ self.normalize_image(x) for x in edges ]

    channels += edges

    return self._get_sliding_window_crops(channels)

  def _get_sliding_window_crops(self, channels):
    width = channels[0].shape[1]
    height = channels[0].shape[0]

    crops = []

    oversample_px = self.crop_size//4
    slide_px = self.crop_size-oversample_px

    this_y = 0
    while(this_y <= height-self.crop_size):
      this_x = 0
      while(this_x <= width-self.crop_size):
        left = this_x
        right = left + self.crop_size
        top = this_y
        bottom = top + self.crop_size

        crop_channels = [ channel[top:bottom, left:right] for channel in channels ]

        crops.append({
          'image': np.stack(crop_channels, axis=-1),
          'coords': (left, right, top, bottom )
        })

        if width-right < self.crop_size and width-right > 0:
          this_x = width-self.crop_size
        else: 
          this_x += slide_px
      if height-bottom < self.crop_size and height-bottom > 0:
        this_y = height-self.crop_size
      else: 
        this_y += slide_px

    return crops

class Combined(MNModel):
  def __init__(self):
    super().__init__()

    self.models = [
      MNModel.get_model("FocalLoss"),
      MNModel.get_model("FocalLoss4Class"),
      MNModel.get_model("FocalLoss4Class2"),
      MNModel.get_model("FocalLoss4Class3")
    ]

    self.model_url = None

  def _load_model(self):
    return True

  def predict_mn(self, img, use_argmax=True, skip_opening=False, threshold=0.1, **kwargs):
    combined = np.zeros_like(img[...,0])
    combined_raw = []
    for model in self.models:
      mn_labels, mn_raw = model.predict_mn(img, use_argmax, skip_opening, **kwargs)
      combined[(mn_labels > 0)] += 1
      combined_raw.append(mn_raw)

    combined[(combined > (threshold*len(self.models)))] = 1
    combined_labels = label(combined, connectivity=1)

    combined_raw = np.mean(combined_raw, axis=0)

    return combined_labels, combined_raw

class FocalLoss(MNModel):
  def __init__(self):
    self.bg_thresh = 143
    self.fg_thresh = 130
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/FocalLoss.tar.gz'

    super().__init__()

class FocalLoss4Class(MNModel):
  def __init__(self):
    self.bg_thresh = 149
    self.fg_thresh = 143
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/FocalLoss4Class.tar.gz'

    super().__init__()

class FocalLoss4Class2(MNModel):
  def __init__(self):
    self.bg_thresh = 251
    self.fg_thresh = 226
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/FocalLoss4Class2.tar.gz'

    super().__init__()

class FocalLoss4Class3(MNModel):
  def __init__(self):
    self.bg_thresh = 233
    self.fg_thresh = 229
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/FocalLoss4Class3.tar.gz'

    super().__init__()


class DiffSobel(MNModel):
  def __init__(self):
    self.bg_thresh = 109
    self.fg_thresh = 122
    self.model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/DiffSobel.tar.gz'

    super().__init__()

  def _get_custom_metrics(self):
    return { 
      'dice_coef': self._get_model_metric('dice_coef'), 
      'tversky': self._get_model_metric('tversky') 
    }

  def _get_mn_predictions(self, img):
    tensors = []
    coords = []
    crops = self._get_image_crops(img)

    for crop in crops:
      sobel_diff = crop['image'][...,2]-10*crop['image'][...,3]
      sobel_diff[sobel_diff < 0] = 0
      tensors.append(tf.convert_to_tensor(
        np.stack([ crop['image'][...,0], crop['image'][...,2], crop['image'][...,3], sobel_diff ], axis=-1)
      ))
      coords.append(crop['coords'])

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset_batchs = dataset.batch(self.batch_size)
    predictions = self.trained_model.predict(dataset_batchs)

    return coords, dataset, predictions

  def predict_mn(self, img, use_argmax=False, skip_opening=False, **kwargs):
    if len(img.shape) <= 2 or img.shape[2] < 2:
      raise ValueError("DiffSobel requires an image with 2 channels")

    return super().predict_mn(img, use_argmax=use_argmax, skip_opening=skip_opening)

class IncorrectDimensions(Exception):
  "Images must be (x,y,c) or (x,y)"
  pass

class ModelNotFound(Exception):
  "That model could not be found"
  pass

class ModelNotLoaded(Exception):
  "That model could not be loaded"
  pass