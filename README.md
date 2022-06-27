
# VAT - PyTorch

A collection of VAT (Virtual Adversarial Training) methods, in PyTorch.

## Install

```bash
$ pip install vat-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/vat-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/vat-pytorch/) 


## Usage 

### Extract Model
The first thing we have to do is extract the chunk of the model that we want to perturb adversarially. A generic example with Huggingface's RoBERTa for sequence classification is as follows, where we can choose the start layer: 

```py 
import torch.nn as nn 
from transformers import AutoModelForSequenceClassification


class ExtractedRoberta(nn.Module):

    def __init__(self):
        super().__init__()
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base')
        self.roberta = model.roberta
        self.layers = model.roberta.encoder.layer  
        self.classifier = model.classifier 
        self.attention_mask = None 
        self.from_layer: int = 0 

    def forward(self, hidden, with_hidden_states = False):
        hidden_states = [] 
        
        for layer in self.layers[self.from_layer:]:
            hidden = layer(hidden, attention_mask = self.attention_mask)[0]
            hidden_states += [hidden]

        logits = self.classifier(hidden)

        return (logits, hidden_states) if with_hidden_states else logits 

    def get_embeddings(self, input_ids):
        return self.roberta.embeddings(input_ids)

    def set_attention_mask(self, attention_mask):
        self.attention_mask = self.roberta.get_extended_attention_mask(
            attention_mask, 
            input_shape = input_ids.shape, 
            device = input_ids.device
        ) # (b, 1, 1, d) 

    def set_from_layer(self, layer: int):
        self.from_layer = layer 
```

### SMART 

<a href="https://aclanthology.org/2020.acl-main.197/">Paper</a>

```py
import torch.nn as nn  
import torch.nn.functional as F 
from vat_pytorch import SMARTLoss, kl_loss, sym_kl_loss

class SMARTClassificationModel(nn.Module):
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, extracted_model, weight):
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

<a href="https://arxiv.org/abs/2005.08156">Paper</a>

```py
import torch.nn as nn  
import torch.nn.functional as F 
from vat_pytorch import ALICELoss, kl_loss

class ALICEClassificationModel(nn.Module):
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, extracted_model, weight):
        super().__init__()
        self.model = extracted_model 
        self.weight = weight
        self.vat_loss = ALICELoss(model = extracted_model, virtual_loss_fn = kl_loss)

    def forward(self, input_ids, attention_mask, labels):
        """ input_ids: (b, s), attention_mask: (b, s), labels: (b,) """
        # Get input embeddings 
        embeddings = self.model.get_embeddings(input_ids)
        # Set iteration specific data (e.g. attention mask) 
        self.model.set_attention_mask(attention_mask)
        # Compute logits 
        logits = self.model(embeddings)
        # Compute CE loss  
        ce_loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        # Compute VAT loss
        vat_loss = self.vat_loss(embeddings, logits, labels) 
        # Merge losses 
        loss = ce_loss + self.weight * vat_loss
        return logits, loss
```

### ALICE++

<a href="https://aclanthology.org/2021.paclic-1.40/">Paper</a>

```py 

```


## Citations

```bibtex
@inproceedings{Jiang2020SMARTRA,
  title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
  author={Haoming Jiang and Pengcheng He and Weizhu Chen and Xiaodong Liu and Jianfeng Gao and Tuo Zhao},
  booktitle={ACL},
  year={2020}
}

@article{Pereira2020AdversarialTF,
  title={Adversarial Training for Commonsense Inference},
  author={Lis Kanashiro Pereira and Xiaodong Liu and Fei Cheng and Masayuki Asahara and Ichiro Kobayashi},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.08156}
}

@inproceedings{Pereira2021ALICEAT,
  title={ALICE++: Adversarial Training for Robust and Effective Temporal Reasoning},
  author={Lis Kanashiro Pereira and Fei Cheng and Masayuki Asahara and Ichiro Kobayashi},
  booktitle={PACLIC},
  year={2021}
}
```