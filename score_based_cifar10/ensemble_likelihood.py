import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import gc
import io
import os
import time

import math
import numpy as np
#import tensorflow as tf
#import tensorflow_gan as tfgan

# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import tensorflow as tf
import tensorflow_gan as tfgan
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from tqdm import tqdm
from torch.distributions.bernoulli import Bernoulli
from itertools import combinations
import random

from ensemble import Ensemble

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt",None,"Checkpoint")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_integer("n_ensembles", 0, "Size of ensemble")
flags.DEFINE_integer("batch_size", 1024, "Sample batch size")
flags.DEFINE_enum("ensemble_type", None, ["none", "mc_dropout", "deep_ensemble", "laplace"], "Type of ensemble")
flags.DEFINE_boolean("consistent_dropout",False,"Decide if Consistent MC dropout or not")
flags.DEFINE_boolean("resume",False,"Resume computation")
flags.DEFINE_integer("num_samples", 50000, "Number of samples")
flags.DEFINE_integer("num_repeats",5,"Number of repeats to compute bpd")
flags.DEFINE_string("checkpoint_dir", None, "Checkpoint folder")
flags.DEFINE_string("nll_folder", None, "The folder name for storing NLL")
flags.mark_flags_as_required(["workdir", "ckpt","config", "batch_size"])

def main(argv):
  n_ensembles = FLAGS.n_ensembles
  ensemble_type = FLAGS.ensemble_type
  n_ensembles = FLAGS.n_ensembles
  consistent_dropout = FLAGS.consistent_dropout
  if (n_ensembles == 0 and ensemble_type != "none"):
      raise ValueError(f"For {ensemble_type}, n_ensembles must be a positive integer.")
  
  workdir = FLAGS.workdir
  eval_folder = FLAGS.eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  config = FLAGS.config
  logging.info(f"Seed: {config.seed}")
  torch.manual_seed(config.seed)
  
  tf.io.gfile.makedirs(eval_dir)

  resume = FLAGS.resume

  batch_size = FLAGS.batch_size
  config.training.batch_size = batch_size
  config.eval.batch_size = batch_size
  
  checkpoint_dir = FLAGS.checkpoint_dir
  nll_folder = FLAGS.nll_folder

  # Initialize the model
  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  sigmas = mutils.get_sigmas(config)
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  
  ensemble = Ensemble(method="arithmetic",ensemble_type=ensemble_type)
 
  if ensemble_type != "deep_ensemble":
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                 decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
               model=score_model, ema=ema)

    # Load checkpoint.
    ckpt = FLAGS.ckpt
    if FLAGS.checkpoint_dir is None:
      checkpoint_dir = "checkpoints"
    else:
      checkpoint_dir = os.path.join("checkpoints",FLAGS.checkpoint_dir)
    ckpt_filename = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    logging.info("Restoring {}".format(ckpt_filename))
    state = restore_checkpoint(ckpt_filename, state, config.device)
    logging.info("Number of training steps: {}".format(state["step"]))
    ema.copy_to(score_model.parameters())
  else:
    score_models = []
    enumerate_ensemble = range(n_ensembles)

    if n_ensembles == 2:
      models_list = list(combinations([0, 1, 2, 3, 4], 2))
      logging.info(models_list)
      enumerate_ensemble = models_list[config.seed]

    for i in enumerate_ensemble:
      score_model = mutils.create_model(config)
      optimizer = losses.get_optimizer(config, score_model.parameters())
      ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)

      # Load checkpoint.
      ckpt = FLAGS.ckpt
      offset = 1 
      checkpoint_dir = os.path.join("checkpoints",f"{ensemble_type}_num_{offset + i}")
      ckpt_filename = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
      logging.info("Restoring: {}".format(ckpt_filename))
      state = restore_checkpoint(ckpt_filename, state, config.device)
      logging.info("checkpoint step: {}".format(state["step"]))
      ema.copy_to(score_model.parameters())
      score_models.append(score_model)


  if ensemble_type == "mc_dropout":
    logging.info("Dropout p={} enabled during inference.".format(config.model.dropout))
    logging.info("Size of ensemble = {}.".format(n_ensembles))
    dropout_rate = config.model.dropout
    shape = config.data.image_size
    # enable_dropout will be activated when calling get_score_fn with ensemble_type in argument
  elif ensemble_type == "deep_ensemble":
    logging.info("No dropout during inference.")
    logging.info("Size of ensemble = {}.".format(n_ensembles))
  else:
    logging.info("No ensemble and no dropout.")

    
  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = FLAGS.num_repeats
    logging.info(f"Test set: {len(ds_bpd)} batchs")
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")



  if ensemble_type == "none":
    nll_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}")
  else:
    nll_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_{ensemble_type}_{n_ensembles}_arithmetic_small_scale"
            + (f"_seed_{config.seed}" if n_ensembles == 2 else ""))
  if nll_folder is not None:
    nll_dir = os.path.join(eval_dir, nll_folder)
  tf.io.gfile.makedirs(nll_dir)
  logging.info(f"Save path: {nll_dir}")

  # Build the likelihood computation function when likelihood is enabled
  config.eval.enable_bpd = True
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, ensemble=ensemble)
    # Compute log-likelihoods (bits/dim) if enabled
  num_samples = FLAGS.num_samples
  num_steps = math.ceil(num_samples / batch_size)

  start_repeat = 0
  start_batch = 0
  repeat_checkpoint_file = os.path.join(nll_dir, f"_repeat_nll_seed_{config.seed}.npz")
  if os.path.exists(repeat_checkpoint_file) and resume:
    logging.info(f"We resume the computation from {repeat_checkpoint_file}")
    checkpoint_data = np.load(repeat_checkpoint_file)
    start_repeat = int(checkpoint_data["repeat"]) # + 1
    start_batch = int(checkpoint_data["batch"]) + 1
    bpds_start = checkpoint_data["bpds"]

  if config.eval.enable_bpd:

    for repeat in range(start_repeat, bpd_num_repeats):
      logging.info(f"starting repeat: {repeat+1}/{bpd_num_repeats}")
      bpds = []
      bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
      
      if repeat == start_repeat and resume and start_batch > 0:
        logging.info(f"Skipping {start_batch} batches.")
        bpds = bpds_start
        for _ in range(start_batch):
          next(bpd_iter)
      else:
        start_batch = 0
      
      for batch_id in range(start_batch, len(ds_bpd)):
        try:
          batch = next(bpd_iter)
        except StopIteration:
          logging.info("End of dataset")
          break
        logging.info("batch: {}/{}".format(batch_id+1,len(ds_bpd)))

        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        
        if ensemble_type == "mc_dropout":
          if consistent_dropout:
            masks = [Bernoulli(torch.full((batch_size,256,shape,shape), 1-dropout_rate,device=config.device)).sample()/(1-dropout_rate) for _ in range(n_ensembles)]
          else:
            masks = [None]*n_ensembles
          bpd = likelihood_fn(score_model, eval_batch,masks)[0]
        elif ensemble_type == "deep_ensemble" or ensemble_type == "laplace":
          bpd = likelihood_fn(score_models, eval_batch)[0]
        else:
          bpd = likelihood_fn(score_model, eval_batch)[0]
          
        bpd = bpd.detach().cpu().numpy().reshape(-1) # shape = (batch_size,)
        bpds.extend(bpd)
        logging.info(
        "ckpt: {}, repeat: {}, batch: {}, mean bpd: {:5f}".format(ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
        bpd_round_id = batch_id + num_samples * repeat
        np.savez(repeat_checkpoint_file, repeat=repeat, batch=batch_id, bpds=bpds)
          
    
      # Save bits/dim to disk or Google Cloud Storage
      logging.info("saving nlls of round {}".format(repeat))
      with tf.io.gfile.GFile(os.path.join(nll_dir,
                              f"nll_round_{repeat}_seed_{config.seed}.npz"),
                               "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, np.asarray(bpds))
        fout.write(io_buffer.getvalue())

if __name__ == "__main__":
  app.run(main)
