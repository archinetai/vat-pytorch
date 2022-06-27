from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count 

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)


class ALICELoss(nn.Module):
    
    def __init__(
        self,
        model: nn.Module,
        virtual_loss_fn: Callable,
        virtual_loss_last_fn: Callable = None,
        gold_loss_fn: Callable = None, 
        gold_loss_last_fn: Callable = None,
        norm_fn: Callable = inf_norm, 
        alpha: float = 1,
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.model = model 
        self.virtual_loss_fn = virtual_loss_fn
        self.virtual_loss_last_fn = default(virtual_loss_last_fn, virtual_loss_fn)
        self.gold_loss_fn = default(gold_loss_fn, virtual_loss_fn)
        self.gold_loss_last_fn = default(gold_loss_last_fn, virtual_loss_fn)
        self.norm_fn = norm_fn
        self.alpha = alpha
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
     
    def forward(self, embed: Tensor, state: Tensor, labels: Tensor) -> Tensor:
        one_hot_labels = F.one_hot(labels).float()

        virtual_loss = self.get_perturbed_loss(
            embed, 
            state, 
            loss_fn = self.virtual_loss_fn, 
            loss_last_fn = self.virtual_loss_last_fn
        )

        gold_loss = self.get_perturbed_loss(
            embed, 
            state = one_hot_labels, 
            loss_fn = self.gold_loss_fn, 
            loss_last_fn = self.gold_loss_last_fn
        )

        return gold_loss + self.alpha * virtual_loss

    @torch.enable_grad()   
    def get_perturbed_loss(self, embed: Tensor, state: Tensor, loss_fn: Callable, loss_last_fn: Callable):
        noise = torch.randn_like(embed, requires_grad = True) * self.noise_var 
        
        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.model(embed_perturbed) 
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
        