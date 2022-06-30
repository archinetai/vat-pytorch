
# VAT - PyTorch

A collection of VAT (Virtual Adversarial Training) methods, in PyTorch.

## Install

```bash
$ pip install vat-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/vat-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/vat-pytorch/) 


## API 

### SMART 
The <a href="https://aclanthology.org/2020.acl-main.197/">SMART paper</a> proposes to find the noise that maximally perturbs the logits when added to the embedding layer, and to use a loss function to make sure that the perturbed logits are as close as possible to the predicted logits. 

```py 
from vat_pytorch import SMARTLoss, inf_norm 

loss = SMARTLoss(
    model: nn.Module,
    loss_fn: Callable,
    loss_last_fn: Callable = None, 
    norm_fn: Callable = inf_norm, 
    num_steps: int = 1,
    step_size: float = 1e-3, 
    epsilon: float = 1e-6,
    noise_var: float = 1e-5
)
```

### ALICE 

The <a href="https://arxiv.org/abs/2005.08156">ALICE paper</a> is analogous to the SMART paper, but adds an additional term to make sure that the perturbed logits are as close as possible to both the predicted logits *and* the ground truth labels. 

```py 
from vat_pytorch import ALICELoss, inf_norm 

loss = ALICEPPLoss(
    model: nn.Module,
    loss_fn: Callable,
    num_classes: int, 
    loss_last_fn: Callable = None,
    gold_loss_fn: Callable = None, 
    gold_loss_last_fn: Callable = None,
    norm_fn: Callable = inf_norm, 
    alpha: float = 1,
    num_steps: int = 1,
    step_size: float = 1e-3, 
    epsilon: float = 1e-6,
    noise_var: float = 1e-5
)
```

### ALICE++ 

The <a href="https://aclanthology.org/2021.paclic-1.40/">ALICE++ paper</a> is analogous to the ALICE paper, but instead of adding noise to the embedding layer, it picks a random layer from the network at each iteration on which to add the noise. 

```py 
from vat_pytorch import ALICEPPLoss, ALICEPPModule, inf_norm 

loss = ALICEPPLoss(
    model: ALICEPPModule,
    loss_fn: Callable,
    num_classes: int, 
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
)
```

## Usage (Classification)

### Extract Model
The first thing we have to do is extract the chunk of the model that we want to perturb adversarially. A generic example with Huggingface's RoBERTa for sequence classification is given. 

```py 
import torch.nn as nn 
from transformers import AutoModelForSequenceClassification

class ExtractedRoBERTa(nn.Module):

    def __init__(self):
        super().__init__()
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base')
        self.roberta = model.roberta
        self.layers = model.roberta.encoder.layer  
        self.classifier = model.classifier 
        self.attention_mask = None 
        self.num_layers = len(self.layers) - 1 

    def forward(self, hidden, with_hidden_states = False, start_layer = 0):
        """ Forwards the hidden value from self.start_layer layer to the logits. """
        hidden_states = [hidden] 
        
        for layer in self.layers[start_layer:]:
            hidden = layer(hidden, attention_mask = self.attention_mask)[0]
            hidden_states += [hidden]

        logits = self.classifier(hidden)

        return (logits, hidden_states) if with_hidden_states else logits 

    def get_embeddings(self, input_ids):
        """ Computes first embedding layer given inputs_ids """ 
        return self.roberta.embeddings(input_ids)

    def set_attention_mask(self, attention_mask):
        """ Sets the correct mask on all subsequent forward passes """ 
        self.attention_mask = self.roberta.get_extended_attention_mask(
            attention_mask, 
            input_shape = attention_mask.shape, 
            device = attention_mask.device
        ) # (b, 1, 1, s) 
```
The function `set_attention_mask` is used to fix the attention mask for all subsequent forward calls, this is necessary if we want to use a mask using any VAT loss. The parameter `start_layer` in the forward function is necessary only if we are using `ALICEPPLoss` since the loss function needs a way to change the start layer internally. 


### SMART

```py
import torch.nn as nn  
import torch.nn.functional as F 
from vat_pytorch import SMARTLoss, kl_loss, sym_kl_loss

class SMARTClassificationModel(nn.Module):
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, extracted_model, weight = 1.0):
        super().__init__()
        self.model = extracted_model 
        self.weight = weight
        self.vat_loss = SMARTLoss(model = extracted_model, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)

    def forward(self, input_ids, attention_mask, labels):
        """ input_ids: (b, s), attention_mask: (b, s), labels: (b,) """
        # Get input embeddings 
        embeddings = self.model.get_embeddings(input_ids)
        # Set mask and compute logits 
        self.model.set_attention_mask(attention_mask)
        logits = self.model(embeddings)
        # Compute CE loss  
        ce_loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        # Compute VAT loss
        vat_loss = self.vat_loss(embeddings, logits) 
        # Merge losses 
        loss = ce_loss + self.weight * vat_loss
        return logits, loss
```

### ALICE

```py
import torch.nn as nn  
import torch.nn.functional as F 
from vat_pytorch import ALICELoss, kl_loss

class ALICEClassificationModel(nn.Module):
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, extracted_model):
        super().__init__()
        self.model = extracted_model 
        self.vat_loss = ALICELoss(model = extracted_model, loss_fn = kl_loss, num_classes = 2)

    def forward(self, input_ids, attention_mask, labels):
        """ input_ids: (b, s), attention_mask: (b, s), labels: (b,) """
        # Get input embeddings 
        embeddings = self.model.get_embeddings(input_ids)
        # Set iteration specific data (e.g. attention mask) 
        self.model.set_attention_mask(attention_mask)
        # Compute logits 
        logits = self.model(embeddings)
        # Compute VAT loss
        loss = self.vat_loss(embeddings, logits, labels) 
        return logits, loss
```

### ALICE++

```py
import torch.nn as nn  
import torch.nn.functional as F 
from vat_pytorch import ALICEPPLoss, kl_loss

class ALICEPPClassificationModel(nn.Module):
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, extracted_model):
        super().__init__()
        self.model = extracted_model 
        self.vat_loss = ALICEPPLoss(
            model = extracted_model, 
            loss_fn = kl_loss,
            num_layers = self.model.num_layers,
            num_classes = 2 
        )
        
    def forward(self, input_ids, attention_mask, labels):
        """ input_ids: (b, s), attention_mask: (b, s), labels: (b,) """
        # Get input embeddings 
        embeddings = self.model.get_embeddings(input_ids)
        # Set iteration specific data (e.g. attention mask) 
        self.model.set_attention_mask(attention_mask)
        # Compute logits 
        logits, hidden_states = self.model(embeddings, with_hidden_states = True) 
        # Compute VAT loss 
        loss = self.vat_loss(hidden_states, logits, labels) 
        return logits, loss
```

Note that `extracted_model` requires a function with the following signature `forward(self, hidden: Tensor, *, start_layer: int) -> Tensor`, the interface `ALICEPPModule` (`from vat_pytorch import ALICEPPModule`) can be used instead of the `nn.Module` class on the extracted model to make sure that the method is present. 


### Wrapped Model Usage 
Any of the above losses can be used as follows with the extracted model. 
```py 
import torch 
from transformers import AutoTokenizer 

extracted_model = ExtractedRoBERTa()
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# Pick one: 
model = SMARTClassificationModel(extracted_model)
model = ALICEClassificationModel(extracted_model)
model = ALICEPPClassificationModel(extracted_model)
# Compute inputs 
text = ["This text belongs to class 1...", "This text belongs to class 0..."]
inputs = tokenizer(text, return_tensors='pt')
labels = torch.tensor([1, 0]) 
# Compute logits and loss 
logits, loss = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], labels = labels)
# To finetune do this for many steps  
loss.backward() 
```

## Citations

```bibtex
@inproceedings{Jiang2020SMARTRA,
  title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
  author={Haoming Jiang and Pengcheng He and Weizhu Chen and Xiaodong Liu and Jianfeng Gao and Tuo Zhao},
  booktitle={ACL},
  year={2020}
}
```

```bibtex
@article{Pereira2020AdversarialTF,
  title={Adversarial Training for Commonsense Inference},
  author={Lis Kanashiro Pereira and Xiaodong Liu and Fei Cheng and Masayuki Asahara and Ichiro Kobayashi},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.08156}
}
```

```bibtex
@inproceedings{Pereira2021ALICEAT,
  title={ALICE++: Adversarial Training for Robust and Effective Temporal Reasoning},
  author={Lis Kanashiro Pereira and Fei Cheng and Masayuki Asahara and Ichiro Kobayashi},
  booktitle={PACLIC},
  year={2021}
}
```
