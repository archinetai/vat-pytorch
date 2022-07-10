from typing import List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count 
from .utils import default, inf_norm


class ALICEPPModule(nn.Module):
    """ Interface for the model provided to ALICEPPLoss """

    def __init__(self):
        super().__init__() 

    def forward(self, hidden: Tensor, *, start_layer: int) -> Tensor:
        raise NotImplementedError()


class ALICEPPLoss(nn.Module):
    
    def __init__(
        self,
        model: ALICEPPModule,
        num_classes: int,
        loss_fn: Callable,
        num_layers: int,
        max_layer: int = None,
        loss_last_fn: Callable = None,
        gold_loss_fn: Callable = None, 
        gold_loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        alpha: float = 1,
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5,
    ) -> None:
        super().__init__()
        self.model = model 
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.num_layers = num_layers
        self.max_layer = min(default(max_layer, num_layers), num_layers) 
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
        random_layer_id = torch.randint(low = 0, high = self.max_layer, size = (1,))[0]

        virtual_loss = self.get_perturbed_loss(
            hidden = hiddens[random_layer_id], 
            state = state, 
            layer_id = random_layer_id, 
            loss_fn = self.loss_fn,
            loss_last_fn = self.loss_last_fn 
        ) 

        label_loss = self.get_perturbed_loss(
            hidden = hiddens[random_layer_id], 
            state = F.one_hot(labels, num_classes=self.num_classes).float(),
            layer_id = random_layer_id, 
            loss_fn = self.gold_loss_fn,
            loss_last_fn = self.gold_loss_last_fn 
        ) 

        return label_loss + self.alpha * virtual_loss

    @torch.enable_grad()   
    def get_perturbed_loss(
        self, 
        hidden: Tensor, 
        state: Tensor, 
        layer_id: int, 
        loss_fn: Callable, 
        loss_last_fn: Callable
    ):
        noise = torch.randn_like(hidden, requires_grad = True) * self.noise_var 
        
        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed hidden and states 
            hidden_perturbed = hidden + noise 
            state_perturbed = self.model(hidden_perturbed, start_layer = layer_id) 
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
        