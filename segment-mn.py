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
    --model=<name>  The name of the MN classifier to use [default: CBLoss]
    --dna-channel=<int>  Which channel is the DNA/chromatin stain [default: 1]
    --nls-channel=<int>  Which channel is the NLS marker [default: 2]
    --mn-intensity-channel=<int>  Which channel to use for determining intact vs ruptured MN [default: 2]
    --mn-intact-thresh=<float>  The minimum ratio threshold of MN Dendra/PN Dendra for it to be called intact [default: 0.16]
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
from src.mnfinder import MNModel

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
  # '--model': And(len, lambda n: locate("mnfinder." + n) is not None, error="MN classifier not found"),
  '--model': len,
  '--dna-channel': And(Use(int), lambda n: n > 0),
  '--nls-channel': And(Use(int), lambda n: n > 0),
  '--mn-intensity-channel': And(Use(int), lambda n: n > 0),
  '--mn-intact-thresh': Use(float),
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
  image
)

if args['--true-mn-mask']:
  mn_truth = np.array(Image.open(args['--true-mn-mask']))
  if zoom_factor != 1:
    mn_truth = cv2.resize(mn_truth, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)

  mn_true_masks = np.zeros(( mn_truth.shape[0], mn_truth.shape[1] ), dtype=np.uint8)
  mn_true_masks[(mn_truth[...,0] > 0)] = 1
  mn_true_masks[(mn_truth[...,2] > 0)] = 2
  if mn_truth.shape[2] == 4:
    mn_true_masks[(mn_truth[...,3] == 0)] = 0
  mn_df, pred_df, summary_df = trained_model.eval_mn_prediction(mn_true_masks, mn_labels)

  print(summary_df)

cv2.imshow("Nuclei", skimage.color.label2rgb(nuclei_labels, trained_model.normalize_image(image[...,0])))
cv2.imshow("MN", skimage.color.label2rgb(mn_labels, trained_model.normalize_image(image[...,0], use_csbdeep=True)))
cv2.waitKey(0)
exit()






