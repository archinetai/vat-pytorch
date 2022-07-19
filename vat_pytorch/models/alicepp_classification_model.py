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