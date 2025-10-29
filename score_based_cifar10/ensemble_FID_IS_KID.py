import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import gc
import io
import os
import time

import numpy as np

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
from ensemble import Ensemble

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt",None,"Checkpoint")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_integer("n_ensembles", 0, "Number of ensembles")
flags.DEFINE_string("agg", "arithmetic", "Aggregation scheme")
flags.DEFINE_string("scale", "small", "scale of init")
flags.DEFINE_integer("batch_size", 1024, "Sample batch size")
flags.DEFINE_integer("num_samples", 50000, "Number of samples to generate")
flags.DEFINE_enum("ensemble_type", None, ["none", "mc_dropout", "deep_ensemble"], "Type of ensemble")
flags.DEFINE_boolean("consistent_dropout",False,"Decide if Consistent MC dropout or not")
flags.DEFINE_boolean("force_dropout", False, "True to force dropout during inference")
flags.DEFINE_boolean("resume", False, "Resume sampling")
flags.DEFINE_boolean("save_samples", False, "Save samples")
flags.DEFINE_string("checkpoint_dir", None, "Checkpoint folder")
flags.DEFINE_string("fid_dir", None, "FID folder")
flags.DEFINE_float("noise", 0., "Noise added to score")
flags.DEFINE_boolean("bootstrap", False, "Compute confidence interval with bootstrap")
flags.mark_flags_as_required(["workdir", "config", "batch_size","ensemble_type"])

def main(argv):
  
  n_ensembles = FLAGS.n_ensembles
  ensemble_type = FLAGS.ensemble_type
  agg = FLAGS.agg
  scale = FLAGS.scale
  if (n_ensembles == 0 and ensemble_type != "none"):
      raise ValueError(f"For {ensemble_type}, n_ensembles must be a positive integer.")
  workdir = FLAGS.workdir
  eval_folder = FLAGS.eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  config = FLAGS.config
  
  torch.manual_seed(config.seed)
  tf.io.gfile.makedirs(eval_dir)

  batch_size = FLAGS.batch_size
  config.training.batch_size = batch_size
  config.eval.batch_size = batch_size

  num_samples = FLAGS.num_samples
  resume = FLAGS.resume
  save_samples = FLAGS.save_samples
  fid_folder = FLAGS.fid_dir
  force_dropout = FLAGS.force_dropout
  consistent_dropout = FLAGS.consistent_dropout
  bootstrap = FLAGS.bootstrap
  
  noise = FLAGS.noise

  random_seed = 0

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
    logging.info(f"Restoring {ckpt_filename}")
    state = restore_checkpoint(ckpt_filename, state, config.device)
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
      
      checkpoint_dir = os.path.join("checkpoints",f"{ensemble_type}_num_{i}")
      ckpt_filename = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
      logging.info(f"Restoring {ckpt_filename}")
      state = restore_checkpoint(ckpt_filename, state, config.device)
      logging.info("Number of training steps: {}".format(state["step"]))
      ema.copy_to(score_model.parameters())
      score_models.append(score_model)



  # Define sampling shape and sampling function.
  sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
  
  logging.info("Level of noise: {}".format(noise))
  z = torch.randn(sampling_shape,device=config.device)
  ensemble = Ensemble(method=agg,ensemble_type=ensemble_type,noise=noise,z=z,K=n_ensembles)

  if ensemble_type == "mc_dropout" or force_dropout:
    dropout_rate = config.model.dropout
    shape = config.data.image_size
    logging.info("Dropout p={} enabled during inference.".format(dropout_rate))
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  elif ensemble_type == "deep_ensemble":
    logging.info("No dropout during inference.")
    logging.info("Number of ensembles = {}.".format(len(score_models)))
  else:
    logging.info("No ensemble and perhaps no dropout.")
  
  if ensemble_type == "none":
    this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}")
    tf.io.gfile.makedirs(this_sample_dir)
  else:
    this_sample_dir = os.path.join(
                       eval_dir,
                       f"checkpoint_{ckpt}_{ensemble_type}_{n_ensembles}_{agg}_{scale}_scale"
                       + (f"_seed_{config.seed}" if n_ensembles == 2 else "")
                     )
    tf.io.gfile.makedirs(this_sample_dir)
  if fid_folder is not None:
    this_sample_dir = os.path.join(eval_dir, fid_folder)
    tf.io.gfile.makedirs(this_sample_dir)
  logging.info(f"Eval dir: {this_sample_dir}")
  logging.info(f"Sampling: {config.sampling.predictor} and {config.sampling.corrector}")

  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, ensemble)
      
  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  num_sampling_rounds = num_samples // config.eval.batch_size + 1

  def compute_statistics(ensemble_size=0):
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    if ensemble_type == "none":
      logging.info("starting to compute statistics.")
      this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}")
      if fid_folder is not None:
        this_sample_dir = os.path.join(eval_dir, fid_folder)
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
    else:
      logging.info(f"starting to compute statistics for Ensemble size = {ensemble_size}.")
      this_sample_dir = os.path.join(
                       eval_dir,
                       f"checkpoint_{ckpt}_{ensemble_type}_{ensemble_size}_{agg}_{scale}_scale"
                       + (f"_seed_{config.seed}" if n_ensembles == 2 else "")
                     )
      if fid_folder is not None:
        this_sample_dir = os.path.join(eval_dir, fid_folder)
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, f"statistics_*_size_{ensemble_size}.npz"))
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])

    if not inceptionv3:
      all_logits = np.concatenate(all_logits, axis=0)
    all_pools = np.concatenate(all_pools, axis=0)

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]
    logging.info(f"all_logits: {all_logits[:num_samples].shape}, all_pools: {all_pools[:num_samples].shape}, data_pools: {data_pools.shape}")
    # Compute FID/KID/IS on all samples together.
    logging.info("computing IS.")
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits[:num_samples])
    else:
      inception_score = -1
    logging.info("computing FID.")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools[:num_samples])
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools[:num_samples])
    logging.info(f"data_pools: {tf_data_pools.shape}, all_pools: {tf_all_pools.shape}")
    logging.info("computing KID.")
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    logging.info("ckpt-%s --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))
    
    if bootstrap:
        results = {}
        for score in ["fid","kid"]:
            logging.info(f"Computing {score.upper()}-10k confidence interval using bootstrap.")
            score_mean, score_std, lower_score, upper_score, _ = evaluation.bootstrap_score(all_pools[:15823], score=score, n_samples=15823, n_bootstrap=100, replace_ratio=1.)
            logging.info(f"{score.upper()} bootstrapped : {score_mean:.7f} ± {score_std:.7f}")
            logging.info(f"CI 95% : [{lower_score:.7f}, {upper_score:.7f}]")

            results[f"{score}_mean"] = score_mean
            results[f"{score}_std"] = score_std
            results[f"lower_{score}"] = lower_score
            results[f"upper_{score}"] = upper_score

    if ensemble_type == "none":
      report_file = os.path.join(this_sample_dir, f"report_{num_samples}.npz")
    else:
      report_file = os.path.join(
             this_sample_dir,
             f"report_size_{ensemble_size}.npz")
    with tf.io.gfile.GFile(report_file,"wb") as f:
      io_buffer = io.BytesIO()
      if bootstrap:
          np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid, **results)
      else:
          np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())
    
  # Main loop to compute metrics
  start_round = 0
  round_checkpoint_file = os.path.join(this_sample_dir, "round_checkpoint.npz")
  if os.path.exists(round_checkpoint_file) and resume:
    checkpoint_data = np.load(round_checkpoint_file)
    start_round = int(checkpoint_data["round"]) + 1
 
  for r in range(start_round, num_sampling_rounds):
    logging.info("sampling -- ckpt: {}, round: {}/{}".format(ckpt, r+1, num_sampling_rounds))
    
    # Directory to save samples. Different for each host to avoid writing conflicts
    if ensemble_type == "none":
      samples, _ = sampling_fn(score_model)
      samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
      samples = samples.reshape(
        (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
      # Write samples to disk or Google Cloud Storage
      if r == start_round or save_samples:
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

      # Force garbage collection before calling TensorFlow code for Inception network
      gc.collect()
      latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
      # Force garbage collection again before returning to JAX code
      gc.collect()
      # Save latent represents of the Inception network to disk or Google Cloud Storage
      with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
        np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
        fout.write(io_buffer.getvalue())
        
    else:
      logging.info("batch: {}/{}, aggregation rule: {}, ensemble size: {}, scale: {}".format(r+1,num_sampling_rounds,agg,n_ensembles, scale))
      if ensemble_type == "mc_dropout":
        if consistent_dropout:
          masks = [1 for _ in range(n_ensembles)]
        else:
          masks = [None]*n_ensembles
        if n_ensembles == 1: r == config.seed 
        samples, _ = sampling_fn(score_model, batch_seed=r, masks=masks)
        
      else: # deep_ensemble assumed
        if agg == "mean_predictions":
          samples = torch.zeros((batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size),device=config.device)
          for k in range(n_ensembles):
            logging.info(f"sample number {k}")
            samples += sampling_fn([score_models[k]], batch_seed=None)[0]
          samples /= n_ensembles
        else:
          samples, _ = sampling_fn(score_models, batch_seed=config.seed) #batch_seed=r
      samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
      samples = samples.reshape(
        (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
      # Write samples to disk or Google Cloud Storage
      if r == start_round or save_samples:
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"samples_{r}_size_{n_ensembles}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())
      
      # Force garbage collection before calling TensorFlow code for Inception network  
      gc.collect()
      latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
      # Force garbage collection again before returning to JAX code
      gc.collect()
        
      # Save latent represents of the Inception network to disk or Google Cloud Storage
      with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"statistics_{r}_size_{n_ensembles}.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
        fout.write(io_buffer.getvalue())
  
    np.savez(round_checkpoint_file, round=r)
  
  if ensemble_type == "none":
    compute_statistics(0)
  else:
    compute_statistics(n_ensembles)
  



if __name__ == "__main__":
  app.run(main)
