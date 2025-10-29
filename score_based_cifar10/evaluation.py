# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing FID/Inception scores."""

#import jax
import torch
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
import kagglehub
import urllib.error
from tqdm import tqdm
from scipy import linalg
INCEPTION_TFHUB = 'https://www.kaggle.com/models/tensorflow/inception/TensorFlow1/tfgan-eval-inception/1' #https://tfhub.dev/tensorflow/tfgan/eval/inception/1
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


#def get_inception_model(inceptionv3=False):
#  if inceptionv3:
#    return tfhub.load(
#      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
#  else:
#    path = kagglehub.model_download("tensorflow/inception/tensorFlow1/tfgan-eval-inception/1") # avoid request http error 403
#    return tfhub.load(INCEPTION_TFHUB)


def get_inception_model(inceptionv3=False):
    if inceptionv3:
        # InceptionV3 standard
        return tfhub.load('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
    else:
        try:
            print("🔍 Trying to load Inception from TensorFlow Hub...")
            model = tfhub.load('https://tfhub.dev/tensorflow/tfgan/eval/inception/1')
            print("✅ Successfully loaded from TensorFlow Hub.")
            return model
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"⚠️ TFHub download failed ({e}). Falling back to KaggleHub...")
            path = kagglehub.model_download("tensorflow/inception/tensorFlow1/tfgan-eval-inception")
            print("✅ Loaded Inception model from KaggleHub.")
            return tfhub.load(path)

def load_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = 'assets/stats/cifar10_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = 'assets/stats/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = torch.cuda.device_count() #jax.local_device_count()
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/GPU: {}' #'/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }

def vectorized_cov(X, axis):

    if X.ndim == 2:
        X = np.expand_dims(X, 0)
        squeeze_result = True
    elif X.ndim == 3:
        squeeze_result = False
    print(X.shape)
    batch_size, n, d = X.shape

    mean = X.mean(axis=axis, keepdims=True)        # (batch_size, 1, features)
    X_centered = X - mean                        # (batch_size, n_samples, features)
    cov = X_centered.transpose(0, 2, 1) @ X_centered / (n - 1)  # (batch_size, features, features)

    return cov[0] if squeeze_result else cov
    

## Available for CIFAR-10 only.

def calculate_activation_statistics(act,axis):

    if act.ndim == 2:
        mu = act.mean(axis=0)  # (features,)
        sigma = np.cov(act,rowvar=False)
    elif act.ndim == 3:
        mu = act.mean(axis=1)  # (batch_size, features)
        sigma = vectorized_cov(act, axis=axis)
    else:
        raise ValueError(f"Invalid shape {act.shape} for activation statistics")
        
    return mu, sigma

def calculate_fid(act1, n_samples=10000, axis=1):

    assets_files = "assets/stats/cifar10_stats.npz"
    act2 = np.load(assets_files)["pool_3"][:50000]

    
    if act1.ndim > 2:
        act1 = np.squeeze(act1)
        if act1.shape[0] == 2048 and act1.shape[1] != 2048:
            act1 = act1.T
    elif act1.ndim == 2 and act1.shape[1] != 2048 and act1.shape[0] == 2048:
        act1 = act1.T
    elif act1.ndim != 2 or act1.shape[1] != 2048:
        raise ValueError(f"Invalid input shape for act1: {act1.shape}")
    

    m1, s1 = calculate_activation_statistics(act1, axis=axis)
    m2, s2 = calculate_activation_statistics(act2, axis=axis)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return np.array([fid_value])


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_kid(act1, n_samples=50000):
    assets_files = "assets/stats/cifar10_stats.npz"
    act2 = np.load(assets_files)["pool_3"][:50000]

    tf_act2 = tf.convert_to_tensor(act2)
    tf_act1 = tf.convert_to_tensor(act1)

    return tfgan.eval.kernel_classifier_distance_from_activations(tf_act2, tf_act1).numpy()

def bootstrap_score(all_pools, score="fid", n_bootstrap=100, n_samples=10000, replace_ratio=0.5):
    
    n_total = len(all_pools)
    assert n_samples <= n_total, "n_samples must be <= total available samples"
    print(n_samples, n_total)
    values = []

    calculate_score = globals().get(f"calculate_{score.lower()}")
        
    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrapping {score.upper()}"):
        if replace_ratio == 1.0:
            indices = np.random.choice(n_total, size=n_samples, replace=True)
        else:
            subset_size = int(replace_ratio * n_samples)
            indices = np.random.choice(n_total, size=subset_size, replace=False)

        all_pools_bootstrap = all_pools[indices]
        value = calculate_score(all_pools_bootstrap, n_samples=all_pools_bootstrap.shape[0])
        values.append(value.item())

    values = np.array(values)
    value_mean = np.mean(values)
    value_std = np.std(values)
    
    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)
    
    return value_mean, value_std, lower, upper, values
