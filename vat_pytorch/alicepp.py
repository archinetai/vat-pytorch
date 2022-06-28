from typing import List, Union, Callable

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


class ALICEPPModule(nn.Module):
    """ Interface for the model provided to ALICEPPLoss """

    def __init__(self):
        super().__init__() 

    def forward(self, hidden: Tensor) -> Tensor:
        raise NotImplementedError()

    def set_start_layer(self, layer: int):
        raise NotImplementedError()


class ALICEPPLoss(nn.Module):
    
    def __init__(
        self,
        model: ALICEPPModule,
        loss_fn: Callable,
        num_layers: int,
        loss_last_fn: Callable = None,
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
        self.num_layers = num_layers 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn) 
        self.gold_loss_fn = default(gold_loss_fn, loss_fn)
        self.gold_loss_last_fn = default(default(gold_loss_last_fn, self.gold_loss_fn), self.loss_last_fn)
        self.norm_fn = norm_fn
        self.alpha = alpha
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
     
    def forward(self, hiddens: List[Tensor], state: Tensor, labels: Tensor) -> Tensor: 

        # Pick random layer on which we apply the perturbation 
        random_layer_idx = torch.randint(low = 0, high = self.num_layers, size = (1,))[0]

        # Set random start layer 
        self.model.set_start_layer(random_layer_idx)

        virtual_loss = self.get_perturbed_loss(
            hidden = hiddens[random_layer_idx], 
            state = state, 
            loss_fn = self.loss_fn,
            loss_last_fn = self.loss_last_fn 
        ) 

        label_loss = self.get_perturbed_loss(
            hidden = hiddens[random_layer_idx], 
            state = F.one_hot(labels).float(), 
            loss_fn = self.gold_loss_fn,
            loss_last_fn = self.gold_loss_last_fn 
        ) 

        # Reset to start layer 
        self.model.set_start_layer(0)

        return label_loss + self.alpha * virtual_loss

    @torch.enable_grad()   
    def get_perturbed_loss(
        self, 
        hidden: Tensor, 
        state: Tensor, 
        loss_fn: Callable, 
        loss_last_fn: Callable
    ):
        noise = torch.randn_like(hidden, requires_grad = True) * self.noise_var 
        
        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed hidden and states 
            hidden_perturbed = hidden + noise 
            state_perturbed = self.model(hidden_perturbed) 
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
        