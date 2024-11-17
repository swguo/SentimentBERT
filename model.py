# model.py

import torch
from torch import nn
from transformers import BertModel

class Classifier(nn.Module):
    def __init__(self, n_classes, model_name):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        o = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(o['pooler_output'])
        return self.out(output)
