from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count 
from .utils import default, inf_norm


class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.model = model 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
     
    @torch.enable_grad()   
    def forward(self, embed: Tensor, state: Tensor):
        noise = torch.randn_like(embed, requires_grad = True) * self.noise_var 
        
        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.model(embed_perturbed) 
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
        