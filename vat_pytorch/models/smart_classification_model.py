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