import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from . import displayer


from tqdm import tqdm


from sklearn import preprocessing, datasets
import dataclasses
from typing import Sequence
import pytorch_warmup as warmup


class DDIM:
  def __init__(self, model, num_diffusion_timesteps = 1000, imgshape = (1,3,256,256), beta_start=0.0001, beta_end = 0.02):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    self.betas = torch.linspace(beta_start, beta_end, steps=self.num_diffusion_timesteps)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas, axis=0))
    self.alphas_cumprod_prev = torch.tensor(np.append(1.0, self.alphas_cumprod[:-1]))
    self.model = model #torch.nn.DataParellel ?
    self.imgshape = imgshape
    
  def instant_blur(self,x0, t):
    """Applies the cumulative blurring to X_0 using the coefficients of timestep t
    Parameters:
    + X, tensor of shape Nxd
    + t, integer, in [0,T-1]
    """
    # Sample some noise from a unit gaussian
    zt = torch.randn(self.imgshape, device=x0.device)

    xt = torch.sqrt(self.alphas_cumprod[t])*x0 + torch.sqrt(1-self.alphas_cumprod[t])*zt
    return xt

  def get_eps_from_UNET(self, x, t, **kwargs):
    # the model outputs:
    # - an estimation of the noise eps (chanels 0 to 2)
    # - learnt variances for the posterior  (chanels 3 to 5)
    # (see Improved Denoising Diffusion Probabilistic Models
    # by Alex Nichol, Prafulla Dhariwal
    # for the parameterization)
    # We discard the second part of the output for this practice session.
    model = kwargs.get("model",self.model)
    mask_seed = kwargs.get("mask_seed",None)
    if mask_seed is not None:
      model.module.update_dropout_seed(mask_seed)
 
    if self.imgshape[-1] == 32: #cifar10
      return model(x,x.new_ones([x.shape[0], ], dtype=torch.long,device=x.device) * t)
    elif self.imgshape[-1] == 256: #ffhq
      return model(x, torch.tensor(t,device=x.device).unsqueeze(0).repeat(self.imgshape[0]))[:,:3,:,:]
      #return model(x, torch.tensor(t,device=x.device).unsqueeze(0))[:,:3,:,:]
        # we repeat batch_size times, else dataparallel does not gather the results from each gpus
    else:
      raise NotImplementedError

      
      
  def predict_xstart_from_eps(self, x, eps, t):
    
    x_start = (
        torch.sqrt(1.0 / self.alphas_cumprod[t])* x
        - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def loss_ffhq(self, x0, criterion=None, flatten = False, **kwargs):
    
    agg = kwargs.get("agg","arithmetic")
    K = kwargs.get('K',0)
    
    t = torch.randint(self.num_diffusion_timesteps, size=(), device=x0.device).item()
    
    true_noise = torch.randn_like(x0,device=x0.device)
   
    xt = torch.sqrt(self.alphas_cumprod.to(x0.device)[t]) * x0 
    + torch.sqrt(1-self.alphas_cumprod.to(x0.device)[t]) * true_noise

    if K == 0 or agg is None:
      predicted_noise = self.model(xt, torch.tensor(t,device=xt.device).unsqueeze(0))[:,:3,:,:]
    else: # deep ensemble only
      eps_stack = torch.stack([model(xt, torch.tensor(t,device=xt.device).unsqueeze(0))[:,:3,:,:] for model in self.model[:K]])
      #print(eps_stack.shape) 
      if agg == "arithmetic":
        predicted_noise = torch.mean(eps_stack, dim=0)
      elif agg == "geometric":
        predicted_noise = torch.exp(
          torch.mean(
            torch.log(
                eps_stack + torch.abs(eps_stack.min())
                + 1e-6  # avoid log(0)
            ),
          dim=0
          )
        ) - torch.abs(eps_stack.min())
      elif agg == "hinton":
        predicted_noise = torch.sum(eps_stack, dim=0)
      elif agg == "median":
        predicted_noise = torch.median(eps_stack, dim=0).values
      elif agg == "max_arg_values":
        max_indices = torch.argmax(torch.abs(eps_stack), dim=0)
        predicted_noise = eps_stack.gather(0, max_indices.unsqueeze(0)).squeeze(0)   
 
    if flatten:
      predicted_noise = predicted_noise.view(predicted_noise.size(0), -1)  # [batch_size, features]
      true_noise = true_noise.view(true_noise.size(0), -1)  # [batch_size, features]
      
    if criterion is not None:
      loss = criterion(predicted_noise, true_noise)
    else:
      #print(predicted_noise.shape,true_noise.shape)
      loss = F.mse_loss(predicted_noise, true_noise, reduction="sum")
    
    return loss
  
  def loss_cifar10(self, x0, criterion=None):
    t = torch.randint(self.num_diffusion_timesteps, size = (x0.shape[0],), device=x0.device)
    #t = torch.full(size=(x0.shape[0],), fill_value=998, dtype=torch.long).to(x0.device) 
    true_noise = torch.randn_like(x0,device=x0.device)
  
    # Compute the blurred sample
    xt = torch.sqrt(self.alphas_cumprod.to(x0.device)[t]).view(-1,1,1,1) * x0 
    + torch.sqrt(1-self.alphas_cumprod.to(x0.device)[t]).view(-1,1,1,1) * true_noise
    #xt = self.instant_blur(x0,t)
    # print(x0.shape, xt.shape)
    # Predict the noise with the neural network
    predicted_noise = self.model(xt,t)
    
    if criterion is not None:
      loss = criterion(predicted_noise, true_noise)
    else:
      loss = F.mse_loss(predicted_noise, true_noise)
    return loss
    
  def sample_image(self, show_steps=True, save_images=True, **kwargs):
    images = []
    with torch.no_grad():  # avoid backprop wrt model parameters
      seed = kwargs.get("seed",None)
      device = kwargs.get("device", "cuda:0")
      gen = torch.Generator(device=device)
      
      if seed is not None:
        gen.manual_seed(seed)
      else:
        gen = None
      xt = torch.randn(self.imgshape,device=device,generator=gen)  # initialize x_t for t=T
      eta = kwargs.get("eta", 0)
      
      agg = kwargs.get("agg","arithmetic") 
      K = kwargs.get('K',0)
      if K < 1:
        if agg == "mean_predictions":
          index = kwargs.get('index',None)
        else:
          agg = None
      else:
        print(f"K = {K}.")
        if isinstance(self.model, list):
            assert K <= len(self.model), f"Argument K (equal to {K}) exceeds the size of the list of models (equal to {len(self.model)})."
        
          

      skip_type = kwargs.get("skip_type","none")
      seq = self.reversed_time_steps
      if skip_type == "uniform":
        S = kwargs.get("num_diffusion_timesteps",100)
        skip = self.num_diffusion_timesteps // S
        seq = np.linspace(0, 1, S) * (self.num_diffusion_timesteps-1)
        seq = [int(s) for s in seq][::-1]
        #seq = np.asarray(list(range(0,self.num_diffusion_timesteps,skip)))[::-1]
      if skip_type == "quad":
        S = kwargs.get("num_diffusion_timesteps",100)
        seq = (np.linspace(0, np.sqrt(self.num_diffusion_timesteps * 0.8), S)** 2)
        seq = [int(s) for s in seq][::-1]

      next_seq = seq[1:] + [0]

      if show_steps:
          seq_iterator = seq
      else:
          seq_iterator = tqdm(seq, desc="Sampling Progress")
      for i, t in enumerate(seq_iterator):
        if skip_type in ["uniform","quad"]:
            next_t = next_seq[i]
        else:
            next_t = t-1
          
        z = torch.randn_like(xt,device=device)

        alpha_t = self.alphas_cumprod[t] # alpha_t
        alpha_t_next = self.alphas_cumprod[next_t] # alpha_{t-1}  
        sigma_t = eta * torch.sqrt((1 - alpha_t / alpha_t_next) * (1 - alpha_t_next) / (1 - alpha_t))
        
        if agg is None:
          eps_t_model = self.get_eps_from_UNET(xt,t)
        elif agg == "mixture_of_experts" and seed is not None:
          gen.manual_seed(seed)
          random_index = torch.randint(K, (1,), generator=gen, device=xt.device).item()
          if isinstance(self.model,list):
            eps_t_model = self.get_eps_from_UNET(xt, t, model = self.model[random_index])
          else:
            eps_t_model = self.get_eps_from_UNET(xt, t, mask_seed = random_index)
        elif agg == "random_select":
          random_index = torch.randint(K, (1,)).item()
          if isinstance(self.model,list):
            eps_t_model = self.get_eps_from_UNET(xt, t, model = self.model[random_index])
          else:
            eps_t_model = self.get_eps_from_UNET(xt, t, mask_seed = random_index)
        elif agg == "mean_predictions":
          if isinstance(self.model,list):
            eps_t_model = self.get_eps_from_UNET(xt, t, model = self.model[index]) 
          else:
            eps_t_model = self.get_eps_from_UNET(xt, t, mask_seed = index)  
        else:
          if isinstance(self.model,list):
            eps_stack = torch.stack([self.get_eps_from_UNET(xt, t, model=model) for model in self.model[:K]]) # K models
          else:
            eps_stack = torch.stack([self.get_eps_from_UNET(xt, t) for _ in range(K)]) # MC dropout
            
        if agg is None or agg in ["mixture_of_experts","random_select","mean_predictions"]:
          pass
        elif agg == "arithmetic":
          eps_t_model = torch.mean(eps_stack, dim=0)
        elif agg == "geometric":
          eps_t_model = torch.exp(
            torch.mean(
              torch.log(
                eps_stack + torch.abs(eps_stack.min())
                + 1e-6  # avoid log(0)
              ),
            dim=0
            )
          ) - torch.abs(eps_stack.min())
        elif agg == "hinton":
          eps_t_model = torch.sum(eps_stack, dim=0)
        elif agg == "median":
          eps_t_model = torch.median(eps_stack, dim=0).values
        elif agg == "max_values":
          eps_t_model = torch.max(torch.abs(eps_stack), dim=0).values
        elif agg == "max_arg_values":
          max_indices = torch.argmax(torch.abs(eps_stack), dim=0)
          eps_t_model = eps_stack.gather(0, max_indices.unsqueeze(0)).squeeze(0)
        elif agg == "deviation":
          eps_t_model = torch.mean(eps_stack, dim=0)
          z = torch.mean(torch.stack([torch.randn_like(xt, device=device) for _ in range(K)]), dim=0)
        else:
          raise NotImplementedError  
        
        x0_pred= self.predict_xstart_from_eps(xt, eps_t_model, t)

        if t > 1:  
          mean = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt((1 - alpha_t_next) - sigma_t ** 2) * eps_t_model
          noise = sigma_t * z
        else:
          mean = x0_pred
          noise = torch.zeros_like(xt) # sigma_1 = 0 since alpha_0 = 1

        xt = mean + noise
        

        if show_steps and (i%(len(seq)//10)==0 or i == len(seq)-1):
          print('Iteration :', i+1, 'Timestep :', t)
          pilimg = displayer.display_as_pilimg(torch.cat((xt, x0_pred), dim=3))
        
        if save_images and i%10==0:
          pilimg = displayer.display_as_pilimg(torch.cat((xt, x0_pred), dim=3),disp=False)
          images.append(pilimg)

    return(xt,images)

  def sample_interpolation(self, show_steps=True, **kwargs):
    def slerp(z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
        )
    with torch.no_grad():  # avoid backprop wrt model parameters
        
        xt_1 = torch.randn(self.imgshape,device=device)  # initialize x_t for t=T
        xt_2 = torch.randn(self.imgshape,device=device)
        xt_ = []
        alpha = torch.arange(0.0, 1.01, 0.1).to(device) 
        for i in range(alpha.size(0)):
            xt_.append(slerp(xt_1, xt_2, alpha[i]))
        xt = torch.cat(xt_, dim=0)

        skip_type = kwargs.get("skip_type","none")
        seq = self.reversed_time_steps
        if skip_type == "uniform":
          S = kwargs.get("num_diffusion_timesteps",100)
          skip = self.num_diffusion_timesteps // S
          seq = np.linspace(0, 1, S) * (self.num_diffusion_timesteps-1)
          seq = [int(s) for s in seq][::-1]
          #seq = np.asarray(list(range(0,self.num_diffusion_timesteps,skip)))[::-1]
        if skip_type == "quad":
          S = kwargs.get("num_diffusion_timesteps",100)
          seq = (np.linspace(0, np.sqrt(self.num_diffusion_timesteps * 0.8), S)** 2)
          seq = [int(s) for s in seq][::-1]

        next_seq = seq[1:] + [0]
        for i, t in enumerate(seq):
          if skip_type in ["uniform","quad"]:
            next_t = next_seq[i]
          else:
            next_t = t-1
          
          z = torch.randn_like(xt, device=device)

          alpha_t = self.alphas_cumprod[t] # alpha_t
          alpha_t_next = self.alphas_cumprod[next_t] # alpha_{t-1}  
          sigma_t = 0.
            
          eps_t_model = self.get_eps_from_UNET(xt,t)
          x0_pred= self.predict_xstart_from_eps(xt, eps_t_model, t)

          if t > 1:  
            mean = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt((1 - alpha_t_next) - sigma_t ** 2) * eps_t_model
          else:
            mean = x0_pred

          xt = mean
          
          if show_steps and (i%(len(seq)//10)==0 or i == len(seq)-1):
              print('Iteration :', i+1, 'Timestep :', t)
              
        pilimg = displayer.display_as_pilimg(xt.permute(1, 2, 0, 3).reshape(3, 256, -1).unsqueeze(0))
        return xt,pilimg
