from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import gc
import io
import os
import time
import sys

import numpy as np
import yaml
#import tensorflow as tf
#import tensorflow_gan as tfgan

# Keep the import below for registering all model definitions
from models.cifar.model import UNet
from models.ffhq.unet import create_model
from models.ffhq import dist_util

import tensorflow as tf
import tensorflow_gan as tfgan
import evaluation
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.distributions.bernoulli import Bernoulli

from diffusion.Diffusion import DDIM
from utils import enable_dropout

from itertools import combinations

import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "config file name")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt",None,"Checkpoint")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("fid_folder", None, "This folder name for storing FID results")
flags.DEFINE_integer("n_ensembles", 0, "Number of ensembles")
flags.DEFINE_integer("batch_size", 1024, "Sample batch size")
flags.DEFINE_enum("ensemble_type", "none", ["none", "mc_dropout", "deep_ensemble", "laplace"], "Type of ensemble")
flags.DEFINE_string("agg", "arithmetic", "Type of aggregation")
flags.DEFINE_boolean("consistent_dropout",False,"Decide if Consistent MC dropout or not")
flags.DEFINE_boolean("force_dropout", False, "True to force dropout during inference")
flags.DEFINE_integer("num_sampling_timesteps", 100, "Number of sampling steps")
flags.DEFINE_integer("num_samples", 50000, "Number of samples")
flags.DEFINE_float("eta", 0., "eta")
flags.DEFINE_integer("seed", 0, "seed for ensemble of size 2")
flags.DEFINE_boolean("resume", False, "Resume sampling or not")
flags.mark_flags_as_required(["workdir", "config", "batch_size"])

def main(argv):
  
  #torch.use_deterministic_algorithms(True)
  
  ensemble_type = FLAGS.ensemble_type
  n_ensembles = FLAGS.n_ensembles
  agg = FLAGS.agg
  if (n_ensembles == 0 and ensemble_type != "none"):
      raise ValueError(f"For {ensemble_type}, n_ensembles must be a positive integer.")
  
  workdir = FLAGS.workdir
  eval_folder = FLAGS.eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  
  model_config_name = FLAGS.config
  with open(os.path.join(workdir, model_config_name), "r") as file:
      model_config = yaml.safe_load(file)
  
  num_samples = FLAGS.num_samples
  ckpt = FLAGS.ckpt
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  
  torch.manual_seed(0)
  tf.io.gfile.makedirs(eval_dir)

  batch_size = FLAGS.batch_size
  num_sampling_timesteps = FLAGS.num_sampling_timesteps
  skip_type = "uniform" if num_sampling_timesteps < 1000 else None

  force_dropout = FLAGS.force_dropout
  consistent_dropout = FLAGS.consistent_dropout
  
  resume = FLAGS.resume

  device = "cuda"
  
  yaml_name = os.path.basename(model_config_name)
  dataset_name = os.path.splitext(yaml_name)[0]    
  if ensemble_type not in ["deep_ensemble","laplace"]:
    checkpoint_file = f"{ckpt}.pt"
    if dataset_name == "cifar10":
      model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], attn=model_config["attn"],
                num_res_blocks=model_config["num_res_blocks"], dropout=model_config["dropout"])
      model = model.to(device)
      if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
         
      logging.info(f"Loading {checkpoint_file} ...")
        
      model.load_state_dict(torch.load(os.path.join(
            checkpoint_dir, checkpoint_file), map_location=device))
    elif dataset_name == "ffhq":
      logging.info(f"Loading {checkpoint_file} ...")
      model_config["model_path"] = os.path.join(checkpoint_dir,checkpoint_file) # load checkpoint ffhq
      #model_config["dropout"] = 0.1
      model = create_model(**model_config)
      model = model.to(device)
      if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    else:
      raise NotImplementedError
    
    
    model.eval();
    if ensemble_type == "mc_dropout": enable_dropout(model)
    
    model_fn = model
    
  else:
    model_fn = []

    if ensemble_type == "laplace":
      generator_laplace = torch.Generator(device=device)
      sigma = torch.load(os.path.join(checkpoint_dir,f"hessian_{dataset_name}_0.pt"))/50000
    
    enumerate_ensemble = range(n_ensembles)
    if n_ensembles == 2:
      seed = FLAGS.seed
      models_list = list(combinations([0, 1, 2, 3], 2))
      logging.info(models_list)
      enumerate_ensemble = models_list[seed]
    
    for k in enumerate_ensemble:

      if FLAGS.ckpt is not None:
        checkpoint_file = f"{FLAGS.ckpt}.pt" # if ckpt in argument
      else:
        if ensemble_type == "laplace":
          vec_params = torch.load(os.path.join(checkpoint_dir,f"{dataset_name}_laplace_params_{k+1}.pt"))
        else: 
          checkpoint_file = f"ckpt_{dataset_name}_num_{k+1}.pt"
        checkpoint_file = f"ckpt_{dataset_name}_num_{k+1}.pt" # if no ckpt in argument
      ckpt = "cifar_ddim_ckpt" if dataset_name == "cifar10" else "ffhq_ddim_ckpt"
        
      
      if dataset_name == "cifar10":
        model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], attn=model_config["attn"],
                num_res_blocks=model_config["num_res_blocks"], dropout=model_config["dropout"])
        model = model.to(device)
        if ensemble_type == "laplace":
          logging.info(f"Putting the vector {dataset_name}_laplace_params_{k+1}.pt into params")
          torch.nn.utils.vector_to_parameters(vec_params, model.parameters())
        if torch.cuda.device_count() > 1:
          logging.info(f"Using {torch.cuda.device_count()} GPUs!")
          model = torch.nn.DataParallel(model)
          
        
        if ensemble_type == "deep_ensemble":
          logging.info(f"Loading {checkpoint_file} ...")
          model.load_state_dict(torch.load(os.path.join(
            checkpoint_dir, checkpoint_file), map_location=device))
        
      elif dataset_name == "ffhq":
        logging.info(f"Loading {checkpoint_file} ...")
        model_config["model_path"] = os.path.join(checkpoint_dir,checkpoint_file)
        model_config["dropout"] = 0.
        model = create_model(**model_config)
        model = model.to(device)
        if ensemble_type == "laplace":
          mean_vec = torch.nn.utils.parameters_to_vector(model.parameters())
          generator_laplace.manual_seed(k)
          z = torch.randn(mean_vec.shape, generator=generator_laplace, device=device)
          logging.info(f"Mean of z: {z.mean()}")
          alpha = 5e-4
          new_params = mean_vec + alpha*torch.sqrt(sigma)*z
          new_model = pickle.loads(pickle.dumps(model))
          logging.info("Putting the vector into params")
          torch.nn.utils.vector_to_parameters(new_params, new_model.parameters())
          
        
        if torch.cuda.device_count() > 1:
          logging.info(f"Using {torch.cuda.device_count()} GPUs!")
          model = torch.nn.DataParallel(model)
         
      else:
        raise NotImplementedError
        
      
      if ensemble_type == "laplace":
        new_model.eval()
        model_fn.append(new_model)
      else:
        model.eval()
        model_fn.append(model)
      '''
      if dataset_name == "cifar10":
        model_fn.append(lambda x, t: model(x,x.new_ones([x.shape[0], ], dtype=torch.long,device=device) * t))
      elif dataset_name == "ffhq":
        model_fn.append(lambda x, t: model(x, torch.tensor(t,device=device).unsqueeze(0).repeat(batch_size))[:,:3,:,:])
        # we repeat batch_size times, else dataparallel does not gather the results from each gpus
      else:
        raise NotImplementedError
      '''    

  def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # En MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # En MB
    logging.info(f"Mémoire allouée : {allocated:.2f} MB, Mémoire réservée : {reserved:.2f} MB")

  # Avant suppression
  print_gpu_memory()
  
  
  if ensemble_type == "laplace":
    logging.info("Deleting objects to clean Vram")
    del mean_vec, z, new_params, sigma, model

  print_gpu_memory()
  sampling_shape = (batch_size,
                      3,
                      model_config["image_size"], model_config["image_size"])
  
  
  ddim = DDIM(model_fn,imgshape=sampling_shape)
  
  if ensemble_type == "mc_dropout" or force_dropout:
    dropout_rate = model_config["dropout"]
    logging.info("Dropout p={} enabled during inference.".format(dropout_rate))
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  elif ensemble_type == "deep_ensemble":
    assert n_ensembles == len(model_fn)
    logging.info("Deep ensemble.")
    logging.info("Number of ensembles = {}.".format(n_ensembles))
  elif ensemble_type == "laplace":
    assert n_ensembles == len(model_fn)
    logging.info("Laplace.")
    logging.info("Size of ensemble = {}".format(n_ensembles))
  else:
    logging.info("No ensemble and perhaps no dropout.")

  if ensemble_type == "none":
    this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_size_{num_sampling_timesteps}")
  else:
    suffix = "" if n_ensembles != 2 else f"_seed_{seed}"
    this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_{ensemble_type}_{n_ensembles}_size_{num_sampling_timesteps}_{agg}{suffix}") 
  if FLAGS.fid_folder is not None:
    this_sample_dir = os.path.join(eval_dir, FLAGS.fid_folder)

  tf.io.gfile.makedirs(this_sample_dir)
  logging.info(f"Eval dir: {this_sample_dir}")
  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = model_config["image_size"] >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  num_sampling_rounds = num_samples // batch_size + 1

  def compute_statistics(ensemble_size=0,dataset_name="cifar10"):
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    if ensemble_type == "none":
      logging.info("starting to compute statistics.")
      this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_size_{num_sampling_timesteps}")
      if FLAGS.fid_folder is not None:
        this_sample_dir = os.path.join(eval_dir, FLAGS.fid_folder)
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
    else:
      logging.info(f"starting to compute statistics for Ensemble size = {ensemble_size}.")
      suffix = "" if ensemble_size != 2 else f"_seed_{seed}"
      this_sample_dir = os.path.join(eval_dir, f"checkpoint_{ckpt}_{ensemble_type}_{ensemble_size}_size_{num_sampling_timesteps}_{agg}{suffix}")
      if FLAGS.fid_folder is not None:
        this_sample_dir = os.path.join(eval_dir, FLAGS.fid_folder)
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, f"statistics_*_size_{ensemble_size}.npz"))
    
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])
    if not inceptionv3:
      all_logits = np.concatenate(all_logits, axis=0)[:num_samples] # 50k by default
    all_pools = np.concatenate(all_pools, axis=0)[:num_samples] # 50k by default
    logging.info(f"Number of samples: {all_pools.shape[0]}")
    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(dataset_name)
    data_pools = data_stats["pool_3"][:50000]

    # Compute FID/KID/IS on all samples together.
    logging.info("computing IS.")
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
      inception_score = -1
    logging.info("computing FID.")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    logging.info("computing KID.")
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    logging.info("ckpt-%s --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

    if ensemble_type == "none":
      report_file = os.path.join(this_sample_dir, f"report_{num_samples}.npz")
    else:
      report_file = os.path.join(this_sample_dir, f"report_{num_samples}_size_{ensemble_size}.npz")
    with tf.io.gfile.GFile(report_file,"wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())
    
  # Main loop to compute metrics
  start_round = 0
  round_checkpoint_file = os.path.join(this_sample_dir, "round_checkpoint.npz")
  if os.path.exists(round_checkpoint_file) and resume:
    checkpoint_data = np.load(round_checkpoint_file)
    start_round = int(checkpoint_data["round"])
    if start_round == num_sampling_rounds-1: start_round += 1  
   
  for r in range(start_round,num_sampling_rounds):
    logging.info("sampling -- ckpt: {}, round: {}/{}".format(ckpt, r+1, num_sampling_rounds))
    np.savez(round_checkpoint_file, round=r)
    # Directory to save samples. Different for each host to avoid writing conflicts
    if ensemble_type == "none":

      samples,_ = ddim.sample_image(show_steps=False,
                           save_images=False,
                           eta=0.0,skip_type=skip_type,
                           num_diffusion_timesteps=num_sampling_timesteps,
                           seed=r)
      samples = 0.5 + 0.5 * samples # put in [0,1]
      samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
      samples = samples.reshape(
        (-1, model_config["image_size"], model_config["image_size"], 3))
      # Write samples to disk or Google Cloud Storage
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
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
        fout.write(io_buffer.getvalue())
        
    else:

      logging.info("batch: {}/{}, ensemble size: {}".format(r+1,num_sampling_rounds,n_ensembles))
      
      #eta = 0.2 if agg == "deviation" else FLAGS.eta
      eta = FLAGS.eta
      print(eta)
      seed = r if agg in ["mixture_of_experts","mean_predictions"] else None # seed for the whole sampling process
      
      if agg == "mean_predictions":
        samples = torch.zeros(sampling_shape,device=device)
        for k in range(n_ensembles):
          logging.info(f"sampling number {k+1}")
          samples += ddim.sample_image(show_steps=False,
                           save_images=False,
                           eta=eta,skip_type=skip_type,
                           num_diffusion_timesteps=num_sampling_timesteps,
                           agg=agg,seed=seed,
                           index = k)[0]
        samples /= n_ensembles
      else:   
        samples,_ = ddim.sample_image(show_steps=False,
                                     save_images=False,
                                     eta=eta,skip_type=skip_type,
                                     num_diffusion_timesteps=num_sampling_timesteps,
                                     seed=seed,
                                     K=n_ensembles, agg=agg,
                                     device=device)
 

      samples = 0.5 + 0.5 * samples # put in [0,1]
      samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
      samples = samples.reshape(
        (-1, model_config["image_size"], model_config["image_size"], 3))
      # Write samples to disk or Google Cloud Storage
      if r == start_round:
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
  

  if ensemble_type == "none":
    compute_statistics(0,dataset_name=dataset_name)
  else:
    compute_statistics(n_ensembles,dataset_name=dataset_name)



if __name__ == "__main__":
  app.run(main)
