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
import logging
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

from ensemble import Ensemble

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt",None,"Checkpoint")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_integer("n_ensembles", 0, "Number of ensembles")
flags.DEFINE_integer("batch_size", 1024, "Sample batch size")
flags.DEFINE_string("agg", "arithmetic", "Aggregation scheme")
flags.DEFINE_string("scale", "small", "scale of init")
flags.DEFINE_enum("ensemble_type", None, ["none", "mc_dropout", "deep_ensemble"], "Type of ensemble")
flags.DEFINE_integer("freq_display",1000,"Frequence to display the loss")
flags.DEFINE_float("noise", 0., "Level of score noise")
flags.DEFINE_string("losses_dir", None, "folder to store loss results")
flags.DEFINE_string("checkpoint_dir", None, "Checkpoint folder")
flags.mark_flags_as_required(["workdir", "ckpt", "config", "batch_size"])



def main(argv):

  ensemble_type = FLAGS.ensemble_type
  n_ensembles = FLAGS.n_ensembles
  if (n_ensembles == 0 and ensemble_type != "none"):
      raise ValueError(f"For {ensemble_type}, n_ensembles must be a positive integer.")
  workdir = FLAGS.workdir
  eval_folder = FLAGS.eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  config = FLAGS.config
  torch.manual_seed(config.seed)
  logging.info(f"Seed: {config.seed}")    
  tf.io.gfile.makedirs(eval_dir)

  batch_size = FLAGS.batch_size
  config.training.batch_size = batch_size
  config.eval.batch_size = batch_size
  agg = FLAGS.agg
  noise = FLAGS.noise
  scale = FLAGS.scale

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
    
  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
   




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
      checkpoint_dir = os.path.join("checkpoints","deep_ensemble_num_1")
    else:
      checkpoint_dir = os.path.join("checkpoints",FLAGS.checkpoint_dir)
    ckpt_filename = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    logging.info(f"Loading checkpoint {ckpt_filename}")
    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())
  else:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    k_list = torch.randperm(5, generator=generator)[:n_ensembles] + 1
    logging.info(f"Ensemble list: {k_list}")
    score_models = []
    for i in range(n_ensembles):
      score_model = mutils.create_model(config)
      optimizer = losses.get_optimizer(config, score_model.parameters())
      ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)

      # Load checkpoint.
      ckpt = FLAGS.ckpt
      checkpoint_dir = os.path.join("checkpoints",f"{ensemble_type}_num_{k_list[i]}")
      ckpt_filename = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
      state = restore_checkpoint(ckpt_filename, state, config.device)
      logging.info(f"Loading checkpoint {ckpt_filename}")
      logging.info(f"Step end: {state['step']}")
      ema.copy_to(score_model.parameters())
      score_models.append(score_model)

  # Fonction to calculate the loss.
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  likelihood_weighting = config.training.likelihood_weighting

  reduce_mean = config.training.reduce_mean
  
  if ensemble_type == "mc_dropout":
    logging.info("Dropout p={} enabled during inference.".format(config.model.dropout))
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  elif ensemble_type == "deep_ensemble":
    logging.info("No dropout during inference.")
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  else:
    logging.info("No ensemble and no dropout.")
  logging.info(f"Scale: {scale}")

  ensemble = Ensemble(method=agg,ensemble_type=ensemble_type)
  loss_fn = losses.get_sde_loss_fn(sde, train=False, reduce_mean=reduce_mean,
                              continuous=continuous, likelihood_weighting=likelihood_weighting, ensemble=ensemble)
  logging.info("Score noise level: {}".format(noise))
  all_losses = []
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  freq_display = FLAGS.freq_display

  for i, batch in tqdm(enumerate(eval_iter)):
    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    eval_batch = scaler(eval_batch)
    losses_batch = []

    # Case no ensemble
    if ensemble_type == "none":
      eval_loss = loss_fn(score_model, eval_batch, noise=noise)
      all_losses.append(eval_loss.item())
    elif ensemble_type == "mc_dropout":
      eval_loss = loss_fn(score_model, eval_batch)
      all_losses.append(eval_loss.item())
    else:
      eval_loss = loss_fn(score_models, eval_batch)
      all_losses.append(eval_loss.item())        
        
   
    if (i + 1) % freq_display == 0:
      logging.info("Finished %dth step loss evaluation" % (i + 1))

  # Save loss values to disk or Google Cloud Storage
  
  if ensemble_type == "none":
    all_losses = np.asarray(all_losses)
    mean_loss = all_losses.mean()
    logging.info("Average loss: {}".format(mean_loss))
    if FLAGS.losses_dir is None:
      losses_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_noise_{noise}_true_snr")
    else:
      losses_dir = os.path.join(eval_dir, FLAGS.losses_dir)
  else:
    all_losses = np.asarray(all_losses)
    mean_loss = np.mean(all_losses,axis=0)
    logging.info("Average losses: {}".format(mean_loss))
    losses_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_{ensemble_type}_{n_ensembles}_{agg}_{scale}_scale")
  logging.info(f"Eval dir: {losses_dir}")
    
  tf.io.gfile.makedirs(losses_dir)
  with tf.io.gfile.GFile(os.path.join(losses_dir, f"loss_seed_{config.seed}.npz"), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=mean_loss)
    fout.write(io_buffer.getvalue())


if __name__ == "__main__":
  app.run(main)
