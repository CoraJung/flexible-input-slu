from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import numpy as np
import torch.nn.functional as F

def get_bert(pretrained=True, pretrained_model_name='bert-base-cased'):
    """Initialize a pretrained BERT model"""
    if pretrained:
        bert = BertModel.from_pretrained(pretrained_model_name)
    else:
        configuration = BertConfig.from_pretrained(pretrained_model_name)
        bert = BertModel(configuration)
    return bert

def encode_text(self, idx):
    df_row = self.df.iloc[idx]
    
    intent = df_row['intent_label']
    encoding = self.bert_tokenizer.encode_plus(
        df_row['transcription'],
        add_special_tokens=True,
        return_token_type_ids=False,
        return_tensors='pt'
        )
    
    ret_dict = {'label':intent, 'encoded_text':encoding['input_ids'].flatten(),
                'text_length':encoding['input_ids'].flatten().shape[0]}
    
    return ret_dict

class BertNLU(nn.Module):
    """BERT NLU module"""
    def __init__(self, pretrained=True, args):
        super().__init__()
        self.bert = get_bert(pretrained)
        if args.dataset == "snips":
            num_classes = 6
        elif args.dataset == "fsc"
            num_classes = 31
        # else:
        #     num_classes = 91

        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_text, text_lengths):
        batch_size = input_text.shape[0]
        max_seq_len = input_text.shape[1]
        mask = torch.arange(max_seq_len, device=text_lengths.device)[None,:] < text_lengths[:,None]
        mask = mask.long() # Convert to 0-1
        _, pooled_output = self.bert(input_ids=input_text, attention_mask=mask)
        logits = self.classifier(pooled_output)
        return logits