"""
Reference: https://github.com/microsoft/denoised-smoothing/blob/master/code/certify.py
"""
from time import time
import datetime
import os
import tensorflow as tf
import numpy as np
from . import certification_utils

SKIP = 20
MAX = -1

def perform_certification_test(base_classifier: tf.keras.Model,
  dataset: (np.ndarray, np.ndarray), sigma: float,
  outfile: str = "certification_output/sigma_0.25") -> None:
  """Performs a certification test.

  :param base_classifier: classification model
  :param dataset: dataset consisting of (images, labels)
  :param sigma: the noise level hyperparameter
  :param outfile: output file to serialize the outputs to
  :return: None
  """
  smoothed_classifier = certification_utils.Smooth(base_classifier,
                 10, sigma)
  if not os.path.exists(outfile.split('sigma')[0]):
    os.makedirs(outfile.split('sigma')[0])
  
  with open(outfile, "w") as f:
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f,
      flush=True)
  print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)

  images, labels = dataset
  for i in range(len(images)):
    if i % SKIP != 0:
      continue
    if i == MAX:
      break

    (x, label) = images[i], labels[i]

    before_time = time()
    # certify the prediction of g around x
    prediction, radius = smoothed_classifier.certify(x, 100,
                             10000,
                             0.001,
                             1000)
    after_time = time()
    correct = int(prediction == label)

    time_elapsed = str(
      datetime.timedelta(seconds=(after_time - before_time)))
    
    with open(outfile, "a") as f:
      print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        i, label, prediction, radius, correct, time_elapsed),
        file=f, flush=True)
    print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
      i, label, prediction, radius, correct, time_elapsed),
      flush=True)
