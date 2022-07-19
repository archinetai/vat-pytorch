import torch.nn as nn 
from transformers import AutoModelForSequenceClassification
from .extracted_model import ExtractedModel 

class ExtractedRoBERTa(ExtractedModel):

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