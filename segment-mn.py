# coding=utf-8

"""TensorFlow 2.0 Fowler MN classifier

Usage:
  segment-mn.py <input>... -o <output> [options]
  segment-mn.py (-h | --help)
  segment-mn.py --version

Arguments:
  <input>   Image to classify
  <output>  Directory to save output

Options:
    -h --help  Show this screen
    --version  Show version
    -o FILE --output=FILE  Where to output the mask
    --model=<name>  The name of the MN classifier to use [default: Combined]
    --dna-channel=<int>  Which channel is the DNA/chromatin stain [default: 1]
    --nls-channel=<int>  Which channel is the NLS marker [default: 2]
    --mn-intensity-channel=<int>  Which channel to use for determining intact vs ruptured MN [default: 2]
    --mn-intact-thresh=<float>  The minimum ratio threshold of MN Dendra/PN Dendra for it to be called intact [default: 0.16]
    --stardist-percentile-min=<float>  Percentile minimum for nucleus segmentation normalization [default: 0.0]
    --stardist-percentile-max=<float>  Percentile minimum for nucleus segmentation normalization [default: 99.8] 
    --stardist-probability-thresh=<float>  StarDist probability threshold for nucleus segmentation [default: 0.479071]
    --stardist-overlap-thresh=<float>  StarDist NMS threshold for nucleus segmentation [default: 0.2]
    --true-mn-mask=<str>  Path to a ground truth MN mask
    --true-nuc-mask=<str>  Path to a ground truth nucleus mask
    --magnification=<int>  Magnification of this image [default: 20]
"""

# Import core functions
import os
from pathlib import Path
from tqdm import tqdm
from docopt import docopt
from schema import Schema, And, Or, Use, SchemaError, Optional
import numpy as np
import pandas as pd
import cv2
from pydoc import locate

# Import NN libs
from mnfinder import MNModel

# Import image handling
from tifffile import TiffWriter, TiffFile
import skimage
from skimage.measure import label
from skimage.color import label2rgb
from PIL import Image

MODELS_ROOT = MNModel.models_root

args = docopt(__doc__, version='2.0')
schema = {
  '<input>': [ os.path.exists ],
  '--output': len,
  '--model': And(len, lambda n: locate("mnfinder." + n) is not None, error="MN classifier not found"),
  '--dna-channel': And(Use(int), lambda n: n > 0),
  '--nls-channel': And(Use(int), lambda n: n > 0),
  '--mn-intensity-channel': And(Use(int), lambda n: n > 0),
  '--mn-intact-thresh': Use(float),
  '--stardist-percentile-min': And(Use(float), lambda n: n >= 0.0 and n <= 100.0),
  '--stardist-percentile-max': And(Use(float), lambda n: n >= 0.0 and n <= 100.0),
  '--stardist-probability-thresh': And(Use(float), lambda n: n >= 0.0 and n <= 1.0),
  '--stardist-overlap-thresh': And(Use(float), lambda n: n >= 0.0 and n <= 1.0),
  Optional('--help'): bool,
  Optional('--version'): bool,
  '--true-mn-mask': Or(lambda n: n is None, os.path.exists),
  '--true-nuc-mask': Or(lambda n: n is None, os.path.exists),
  '--magnification': And(Use(int), lambda n: n > 0)
}
try:
  args = Schema(schema).validate(args)
except SchemaError as error:
  print(error)
  exit(1)

model_path = MODELS_ROOT / args['--model']
output_path = Path(args['--output'])

# Channel arguments are 1-indexed. Switch to 0-indexed
args['--dna-channel'] -= 1
args['--nls-channel'] -= 1
args['--mn-intensity-channel'] -= 1

# Collect images into a single TIFF
zoom_factor = 20/args['--magnification']
channels = []
for path in args['<input>']:
  with TiffFile(path) as f:
    if len(f.pages) > 1:
      # Channels are in the first axis
      for page in f.pages:
        channels.append(page.asarray())
    else:
      img = f.pages[0].asarray()
      if len(img.shape) == 3:
        for idx in range(img.shape[2]):
          channels.append(img[...,idx])
      else:
        channels.append(img)

if zoom_factor != 1:
  for idx,channel in enumerate(channels):
    channels[idx] = cv2.resize(channels[idx], None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)

if len(channels) > 1:
  image = np.stack([ channels[args['--dna-channel']], channels[args['--nls-channel']] ], axis=-1)
else:
  image = np.stack([ channels[args['--dna-channel']] ], axis=-1)


# Load MN segmenter
trained_model = MNModel.get_model(args['--model'])

nuclei_labels, mn_labels, mn_raw = trained_model.predict_field(
  image,
  p_min=args['--stardist-percentile-min'], 
  p_max=args['--stardist-percentile-max'],
  prob_thresh=args['--stardist-probability-thresh'],
  nms_thresh=args['--stardist-overlap-thresh']
)

# if args['--true-mn-mask']:
#   mn_truth = np.array(Image.open(args['--true-mn-mask']))

#   mn_true_masks = np.zeros(( mn_truth.shape[0], mn_truth.shape[1] ), dtype=np.uint8)
#   mn_true_masks[(mn_truth[...,0] > 0)] = 1
#   mn_true_masks[(mn_truth[...,2] > 0)] = 2
#   if mn_truth.shape[2] == 4:
#     mn_true_masks[(mn_truth[...,3] == 0)] = 0
#   trained_model.eval_mn_prediction(mn_true_masks, mn_labels)
  
#   intact_mn = np.zeros_like(true_mn_labels)
#   ruptured_mn = np.zeros_like(true_mn_labels)
#   intact_mn[(mn_truth[...,0] == 255)] = 1
#   ruptured_mn[(mn_truth[...,3] == 255)] = 1
#   if mn_truth.shape[2] == 4:
#     intact_mn[(mn_truth[...,3] == 0)] = 0
#     ruptured_mn[(mn_truth[...,3] == 0)] = 0

#   mn_df = {
#     'true_mn_label': [],
#     'intact': [],
#     'found': []
#   }

#   pred_df = {
#     'pred_mn_label': [],
#     'exists': []
#   }

#   summary_df = {
#     'image': [],
#     'num_mn': [],
#     'num_intact_mn': [],
#     'num_ruptured_mn': [],
#     'num_predictions': [],
#     'num_mn_found': [],
#     'num_intact_mn_found': [],
#     'num_ruptured_mn_found': [],
#     'iou': [],
#   }

#   for mn_label in np.unique(true_mn_labels):
#     if mn_label == 0:
#       continue

#     mn_df['true_mn_label'].append(mn_label)
#     if np.sum(intact_mn[(true_mn_labels == mn_label)]) > 0:
#       mn_df['intact'].append(True)
#     else:
#       mn_df['intact'].append(False)

#     if np.sum(mn_labels[(true_mn_labels == mn_label)]) > 0:
#       mn_df['found'].append(True)
#     else:
#       mn_df['found'].append(False)

#   for mn_label in np.unique(mn_labels):
#     pred_df['pred_mn_label'].append(mn_label)
#     if np.sum(true_mn_labels[(mn_labels == mn_label)]) > 0:
#       pred_df['exists'].append(True)
#     else:
#       pred_df['exists'].append(False)

#   mn_df = pd.DataFrame(mn_df)
#   pred_df = pd.DataFrame(pred_df)

#   summary_df['image'].append(args['<input>'][0])
#   summary_df['num_mn'].append(mn_df.shape[0])
#   summary_df['num_intact_mn'].append(np.sum(mn_df['intact']))
#   summary_df['num_ruptured_mn'].append(mn_df.shape[0]-np.sum(mn_df['intact']))
#   summary_df['num_predictions'].append(pred_df.shape[0])
#   summary_df['num_mn_found'].append(np.sum(mn_df['found']))
#   summary_df['num_intact_mn_found'].append(np.sum(mn_df.loc[mn_df['intact'] == True, 'found']))
#   summary_df['num_ruptured_mn_found'].append(np.sum(mn_df.loc[mn_df['intact'] == False, 'found']))

#   intersection = np.sum(np.logical_and((mn_labels > 0), (true_mn_labels > 0)))
#   union = np.sum(np.logical_or((mn_labels > 0), (true_mn_labels > 0)))
#   if union == 0:
#     summary_df['iou'].append(0)
#   else:
#     summary_df['iou'].append(intersection / union)

#   summary_df = pd.DataFrame(summary_df)

#   summary_df['ppv'] = summary_df['num_mn_found']/summary_df['num_predictions']
#   summary_df['recall'] = summary_df['num_mn_found']/summary_df['num_mn']
#   summary_df['intact_recall'] = summary_df['num_intact_mn_found']/summary_df['num_intact_mn']
#   summary_df['ruptured_recall'] = summary_df['num_ruptured_mn_found']/summary_df['num_ruptured_mn']

#   print(summary_df)

cv2.imshow("Nuclei", skimage.color.label2rgb(nuclei_labels, trained_model.normalize_image(image[...,0])))
cv2.imshow("MN", skimage.color.label2rgb(mn_labels, trained_model.normalize_image(image[...,0], use_csbdeep=True)))
cv2.waitKey(0)
exit()






