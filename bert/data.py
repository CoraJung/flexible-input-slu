import torch
import torch.nn as nn
from torch.nn.utils import rnn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from sklearn import preprocessing
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

# Step 1: Read the data
def read_data(data_root):
    data_root = data_root
  
    df_train = pd.read_csv(os.path.join(data_root, 'data/', '{}_data.csv'.format('train')))
    df_val = pd.read_csv(os.path.join(data_root, 'data/', '{}_data.csv'.format('valid')))
    df_test = pd.read_csv(os.path.join(data_root, 'data/', '{}_data.csv'.format('test')))
        
    return df_train, df_val, df_test

#main: 

# Step 2: Encode intent labels
class IntentEncoder(Dataset):
    '''
    Read dataset from args
    Inspired by BaseFluentSpeechDataset(BaseDataset)
    '''
    def __init__(self, df, dataset='snips', intent_encoder=None):
        self.df = df
        self.dataset = dataset
        pretrained_model_name = 'bert-base-cased'
        if self.dataset == 'fsc':
            self.df['intent'] = self.df[['action', 'object', 'location']].apply('-'.join, axis=1)

        if intent_encoder is None:
            intent_encoder = preprocessing.LabelEncoder()
            intent_encoder.fit(self.df['intent'])
        self.intent_encoder = intent_encoder
        self.df['intent_label'] = intent_encoder.transform(self.df['intent'])
        self.labels_set = set(self.df['intent_label'])
        self.label2idx = {}
        for label in self.labels_set:
            idx = np.where(self.df['intent_label'] == label)[0]
            self.label2idx[label] = idx

        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

# Step 3: Call BERT to encode text

def encode_text(idx, bert_tokenizer):
    
    df_row = df.iloc[idx]

    intent = df_row['intent_label']
    encoding = bert_tokenizer.encode_plus(
        df_row['transcription'],
        add_special_tokens=True,
        return_token_type_ids=False,
        return_tensors='pt'
        )

    ret_dict = {'label':intent, 'encoded_text':encoding['input_ids'].flatten(),
                'text_length':encoding['input_ids'].flatten().shape[0]}

    return ret_dict


class BaseDataset(IntentEncoder):
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.encode_text(idx, self.bert_tokenizer)

# Step 4: Put everything into DataLoader
def default_collate_classifier(inputs):
    '''
    Pads and collates into a batch for training
    Returns:
        A dictionary containing
        'text_length': (B,) length of each utterance
        'label': (B,) label of each utterance
        'encoded_text': (B,) encoded text of each utterance
    '''
   
    labels = [data['label'] for data in inputs] #inputs= batch contains X data (=dictionary)
    encoded_text = [data['encoded_text'] for data in inputs]
    text_lengths = [data['text_length'] for data in inputs]
    padded_text = rnn.pad_sequence(encoded_text, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    
    return {'label':labels, 'encoded_text':padded_text, 'text_length':text_lengths}

def get_dataloaders(data_root, batch_size, dataset='snips', num_workers=0, *args, **kwargs):
    df_train, df_val, df_test = read_data(data_root)
    
    
    train_dataset = BaseDataset(df_train, dataset, *args, **kwargs)
    val_dataset = BaseDataset(df_val, dataset, train_dataset.intent_encoder, *args, **kwargs)
    test_dataset = BaseDataset(df_test, dataset, train_dataset.intent_encoder, *args, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_collate_classifier, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=default_collate_classifier, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_collate_classifier, num_workers=num_workers)

    return train_loader, val_loader, test_loader


