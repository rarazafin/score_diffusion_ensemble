import torch
import torch.nn as nn
import numpy as np




class Ensemble():
    def __init__(self, method='arithmetic', ensemble_type='mc_dropout', **kwargs):
        """
        Ensemble class for combining multiple models.

        Parameters:
        - models: list of scores
        - method: str, method for combining predictions.
        - method 
        """
        self.method = method
        self.ensemble_type = ensemble_type
        self.K = kwargs.get('K',1)
        self.noise = kwargs.get('noise',None)
        self.z = kwargs.get('z',None) 
    def set_method(self, method):
        self.method = method
        
    def aggregate_fn(self, score_fn, x, t, masks=None, seed=None):
        
        gen = torch.Generator(device=x.device)

        if self.ensemble_type == "none" and self.noise is not None and self.z is not None and self.noise != 0:
            score = score_fn(x, t)  # shape: (B, ...)
            #z = torch.randn_like(x)

            # Flatten per sample
            B = x.shape[0]
            score_flat = score.view(B, -1)
            z_flat = self.z.view(B, -1)

            score_norm = score_flat.norm(p=2, dim=1, keepdim=True)  # shape: (B, 1)
            z_norm = z_flat.norm(p=2, dim=1, keepdim=True)          # shape: (B, 1)

            eps = 1e-8
            rescale = self.noise * (score_norm / (z_norm + eps))  # shape: (B, 1)

            # Rescale noise and reshape
            new_noise = rescale * z_flat  # shape: (B, D)
            new_noise = new_noise.view_as(x)
            
            return score + new_noise

        if self.ensemble_type == "none":
            return score_fn(x,t)
        
        if self.ensemble_type == "mc_dropout":
            if masks[0] is None:
              if (t[0].item() > 0):
                score_stack = torch.stack([score_fn(x,t) for i in range(len(masks))])
              else:
                return score_fn(x,t)
            else:
              if self.method == "random_select":
                random_index = torch.randint(self.K, (1,)).item()
                return score_fn(x,t,mask_seed=random_index)
              if self.method == "mixture_of_experts":
                gen.manual_seed(seed)
                random_index = torch.randint(self.K, (1,), generator=gen, device=x.device).item()
                return score_fn(x,t,mask_seed=random_index)
              if self.K == 1:
                index = seed
                return score_fn(x,t,mask_seed=index)
              score_stack = torch.stack([score_fn(x,t,mask_seed=i) for i in range(len(masks))])
        elif self.ensemble_type == "deep_ensemble":
            if self.method == "mixture_of_experts":
              gen.manual_seed(seed)
              random_index = torch.randint(self.K, (1,), generator=gen, device=x.device).item()
              return score_fn[random_index](x,t)
            if (t[0].item() > 0):
              score_stack = torch.stack([unique_score_fn(x,t) for unique_score_fn in score_fn])
            else:
              return score_fn[seed](x,t) 
        else:
            raise NotImplementedError("ensemble_type should be mc_dropout, deep_ensemble, or none")
        

        if self.method == "mean_predictions":
            return score_stack[0]
        elif self.method in ['arithmetic','deviation']:
            return torch.mean(score_stack, dim=0)

        elif self.method == 'geometric':

            return torch.exp(
                torch.mean(
                    torch.log(
                        score_stack + torch.abs(score_stack.min())
                        + 1e-6  # avoid log(0)
                        ),
                    dim=0
                    )
            ) - torch.abs(score_stack.min())
        elif self.method == 'median':
            return torch.median(score_stack, dim=0).values    
        elif self.method == 'hinton':
            return torch.sum(score_stack, dim=0)
        
        elif self.method == "max_values":
            return torch.max(torch.abs(score_stack), dim=0).values
        
        elif self.method == "max_arg_values":
            max_indices = torch.argmax(torch.abs(score_stack), dim=0)
            return score_stack.gather(0, max_indices.unsqueeze(0)).squeeze(0)
       
        elif self.method == "random_select":
            random_index = torch.randint(self.K, (1,)).item()
            random_score_fn = score_fn[random_index]
            return random_score_fn(x, t)
         
        elif self.method == "mixture_of_experts" and seed is not None:
            gen.manual_seed(seed)
            random_index = torch.randint(len(score_fn), (1,), generator=gen).item()
            
            random_score_fn = score_fn[random_index]
            return random_score_fn(x, t)
        
        else:
            raise NotImplementedError('Aggregation method not implemented.')
    
    def __call__(self, score_fn, x, t, masks=None, seed=None):
        return self.aggregate_fn(score_fn, x, t, masks, seed)



    def diversity(self, x, t, score_fn, K=5, reduce="no"):
        if self.ensemble_type == "deep_ensemble":
            eps_stack = torch.stack([model(x, t, None) for model in score_fn[:K]]) # K models
        else:
            eps_stack = torch.stack([score_fn(x, t, None) for _ in range(K)]) # MC dropout
      
        combination = torch.mean(eps_stack, dim=0) #  [b,c,h,w]
        diversity_per_image = torch.mean((eps_stack - combination)**2, dim=[0,2,3,4])
        return diversity_per_image if reduce=="no" else torch.mean(diversity_per_image)
