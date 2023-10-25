from transformers import ElectraForPreTraining
from torch import nn
import torch
import torch.nn.functional as F
import sys

class ClassModel(ElectraForPreTraining):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    def freeze_layers(self, num_layers):
        if num_layers is None:
            return
        for name, param in self.electra.named_parameters():
            if name.startswith('embeddings') or int(name.split('.')[2]) < num_layers:
                param.requires_grad = False
    
    def forward(self, input_ids, pred_mode, start_at=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None):
        if start_at and pred_mode != 'inner':
            for i in range(start_at, 12):
                layer_outputs = self.electra.encoder.layer[i](
                                    input_ids,
                                    attention_mask,
                                    head_mask
                                )
                input_ids = layer_outputs[0]
            last_hidden_states = input_ids
        else:
            model_outputs = self.electra(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = model_outputs[0]
        if pred_mode == "inner":
            return model_outputs[1][start_at]
        if pred_mode == "cls":
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode == "prompt":
            logits = self.discriminator_predictions(last_hidden_states)
        else:
            sys.exit("Wrong pred_mode!")
        return logits
