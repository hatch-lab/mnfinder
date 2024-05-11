from .mnfinder import MNSegmenter, MNModelDefaults
from .kerasmodels import SegmenterUNet
import tensorflow as tf
import numpy as np

class DistSegmenter(MNSegmenter):
  model_url = 'https://fh-pi-hatch-e-eco-public.s3.us-west-2.amazonaws.com/mn-segmentation/models/DistSegmenter.tar.gz'

  crop_size = 128

  def _build_model(self, training=False):
    factory = SegmenterUNet()
    return factory.build(self.crop_size, 2, num_output_classes=1, depth=4, training=training)

  def _get_model_metric(self, name):
    sigmoid_focal_crossentropy = super()._get_model_metric('sigmoid_focal_crossentropy')

    def mse_hull(y_true, y_pred):
      return tf.keras.losses.MSE(y_true[...,1], y_pred[...,1])

    def mse_edge(y_true, y_pred):
      return tf.keras.losses.MSE(y_true[...,2], y_pred[...,2])

    def hybrid_loss(y_true, y_pred):
      f_loss = sigmoid_focal_crossentropy(y_true, y_pred)
      hull_loss = mse_hull(y_true, y_pred)
      edge_loss = mse_edge(y_true, y_pred)
      return f_loss+hull_loss+edge_loss


    metrics = { 
      'hybrid_loss': hybrid_loss,
      'mse_hull': mse_hull,
      'mse_edge': mse_edge,
      'sigmoid_focal_crossentropy': sigmoid_focal_crossentropy
    }

    if name is None:
      return metrics

    return metrics[name]

  def _get_loss_function(self):
    return self._get_model_metric('hybrid_loss')