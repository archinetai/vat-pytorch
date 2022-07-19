import torch.nn as nn 

class ExtractedModel(nn.Module):
    """ Interface to be used with extracted models """ 
    
    def forward(self, hidden, *, start_layer = 0):
        """ Forwards the hidden value from self.start_layer layer to the logits. """
        raise NotImplementedError() 

    def get_embeddings(self, input_ids):
        """ Computes first embedding layer given inputs_ids """ 
        raise NotImplementedError() 

    def set_attention_mask(self, attention_mask):
        """ Sets the correct mask on all subsequent forward passes """ 
        # This is optional 