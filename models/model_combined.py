# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
import transformers
from models.layers import SimpleEncoder, SimpleMaxPoolDecoder, SubsampledBiLSTMEncoder, SimpleMaxPoolClassifier, SimpleSeqDecoder, get_bert, MaskedMaxPool, ConvolutionalSubsampledBiLSTMEncoder
import os
import numpy as np

""" Combined model (Alexa & Lugosch) """
import lugosch.models
### Note we need to read config or hard-code depending on what args we are using

class SLUModelBase(nn.Module):
    """Baseclass for SLU models"""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class SLU(SLUModelBase):
    """Baseline SLU model"""
    def __init__(self, input_dim, encoder_dim, num_layers, num_classes):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_layers)
        self.decoder = SimpleMaxPoolDecoder(input_dim=encoder_dim, hidden_dim=encoder_dim, num_classes=num_classes)

    def forward(self, feats, lengths):
        hiddens = self.encoder(feats, lengths)
        logits = self.decoder(hiddens, lengths)
        return logits


class SubsampledSLU(SLUModelBase):
    """Subsampled SLU model"""
    def __init__(self, input_dim, encoder_dim, num_layers, num_classes, decoder_hiddens=[]):
        super().__init__()
        self.encoder = SubsampledBiLSTMEncoder(input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_layers)
        self.decoder = SimpleMaxPoolClassifier(input_dim=2*encoder_dim, num_classes=num_classes, hiddens=decoder_hiddens)

    def forward(self, feats, lengths):
        hiddens, lengths = self.encoder(feats, lengths)
        logits = self.decoder(hiddens, lengths)
        return logits


class Seq2Seq(SLUModelBase):
    """Baseline sequence 2 sequence implementation for SLU"""
    def __init__(self, input_dim, encoder_dim, num_layers, embedding_dim, vocab_size, num_heads):
        super().__init__()
        self.encoder = SubsampledBiLSTMEncoder(input_dim, encoder_dim, num_layers)
        self.decoder = SimpleSeqDecoder(vocab_size, embedding_dim, 2*encoder_dim, num_heads)

    def forward(self, feats, lengths, targets=None, training=False):
        hiddens, lengths = self.encoder(feats, lengths)
        if training:
            predictions = self.decoder(hiddens, hiddens, lengths, targets=targets, training=True)
        else:
            predictions = self.decoder(hiddens, hiddens, lengths, targets=None, training=False)
        return predictions


class BertNLU(nn.Module):
    """BERT NLU module"""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.bert = get_bert(pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
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
    
class FinalPool(torch.nn.Module):
    def __init__(self):
        super(FinalPool, self).__init__()

    def forward(self, input):
        """
        (Lugosch's models.py)
        input : Tensor of shape (batch size, T, Cin)

        Outputs a Tensor of shape (batch size, Cin).
        """

        return input.max(dim=1)[0]

class Downsample(torch.nn.Module):
    
    """
    (Lugosch's models.py)
    Downsamples the input in the time/sequence domain
    """
    def __init__(self, method="none", factor=1, axis=1):
        super(Downsample,self).__init__()
        self.factor = factor
        self.method = method
        self.axis = axis
        methods = ["none", "avg", "max"]
        if self.method not in methods:
            print("Error: downsampling method must be one of the following: \"none\", \"avg\", \"max\"")
            sys.exit()

    def forward(self, x):
        if self.method == "none":
            return x.transpose(self.axis, 0)[::self.factor].transpose(self.axis, 0)
        if self.method == "avg":
            return torch.nn.functional.avg_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)
        if self.method == "max":
            return torch.nn.functional.max_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)

class RNNSelect(torch.nn.Module):
    def __init__(self):
        super(RNNSelect, self).__init__()

    def forward(self, input):
        """
        (Lugosch's models.py)
        input : tuple of stuff

        Outputs a Tensor of shape 
        """
        return input[0] 
    
    
class JointModel(nn.Module):
    """JointModel which combines both modalities"""
    """Replace Alexa's audio embedding with Lugosch's word embeddings"""
    

    def __init__(self,config, input_dim, num_layers, num_classes, encoder_dim=None, bert_pretrained=True, bert_pretrained_model_name='bert-base-cased'):
        super().__init__()
        
        # BERT
        self.bert = get_bert(bert_pretrained, bert_pretrained_model_name)
        self.maxpool = MaskedMaxPool()
        
        # Remove Alexa's encoder 
        
        # Add Lugosch's model 
        self.lugosch_model = lugosch.models.PretrainedModel(config)
        pretrained_model_path = os.path.join(config.libri_folder, "libri_pretraining", "model_state.pth")       
        self.lugosch_model.load_state_dict(torch.load(pretrained_model_path))
        self.config = config

        # freeze phoneme and word layers 
        self.freeze_all_layers()
        self.unfreezing_index = 1
        
        # Lugosch's Intent Module (class Model in models.py)
        self.intent_layers = []
        self.num_values_total = num_classes #default for fluentai
        
        self.num_rnn_layers = len(config.intent_rnn_num_hidden)
        self.out_dim = config.word_rnn_num_hidden[-1]
        if config.word_rnn_bidirectional:
            self.out_dim *= 2 
        for idx in range(self.num_rnn_layers):
            # recurrent
            print("config.intent_rnn_bidirectional :",config.intent_rnn_bidirectional)
            
            layer = torch.nn.GRU(input_size=self.out_dim, hidden_size=config.intent_rnn_num_hidden[idx], batch_first=True, bidirectional=config.intent_rnn_bidirectional)
            layer.name = "intent_rnn%d" % idx
            self.intent_layers.append(layer)

            self.out_dim = config.intent_rnn_num_hidden[idx]
            if config.intent_rnn_bidirectional:
                self.out_dim *= 2

            # grab hidden states of RNN for each timestep
            layer = RNNSelect()
            layer.name = "intent_rnn_select%d" % idx
            self.intent_layers.append(layer)

            # dropout
            layer = torch.nn.Dropout(p=config.intent_rnn_drop[idx])
            layer.name = "intent_dropout%d" % idx
            self.intent_layers.append(layer)

            # downsample
            layer = Downsample(method=config.intent_downsample_type[idx], factor=config.intent_downsample_len[idx], axis=1)
            layer.name = "intent_downsample%d" % idx
            self.intent_layers.append(layer)

            # remove final-classifier

        layer = FinalPool() #maxpool 3D - 2D
        layer.name = "final_pool"
        self.intent_layers.append(layer)

        self.lugosch_intent = torch.nn.ModuleList(self.intent_layers)
        self.aux_embedding = nn.Linear(config.enc_dim, self.bert.config.hidden_size) #bert_hidden_size = 768 enc_dim = 128
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
       

    def forward(self, audio_feats=None, audio_lengths=None, input_text=None, text_lengths=None, text_only=False):
        if text_only:
            return self.forward_text(input_text, text_lengths)
        outputs = {}
        
        if audio_feats is not None:
            
            # print(f"audio feats from JointModel : {audio_feats.size()}")
            #hiddens, lengths = self.speech_encoder(audio_feats, audio_lengths)
            audio_hiddens = self.lugosch_model.compute_features(audio_feats) #audio hiddens is 3D
            lengths = audio_lengths
           
            #pass input through Intent Module (self.lugosch_intent)
            for layer in self.lugosch_intent:
                audio_hiddens = layer(audio_hiddens) #2D output
            print(f"audio_hiddens size (output of intent module): {audio_hiddens.size()}")
            
            # create intent embeddings (named it audio embeddings)
            audio_embeddings = self.aux_embedding(audio_hiddens) #2D align with bert hidden size
            print(f"audio_embedding: {audio_embedding.size()}")
                
            # classify intents
            audio_logits = self.classifier(audio_embeddings)
            print(f"audio logits: {audio_logits.size()}")

            outputs['audio_embed'], outputs['audio_logits'] = audio_embedding, audio_logits

        if input_text is not None:
            batch_size = input_text.shape[0]
            max_seq_len = input_text.shape[1]
            attn_mask = torch.arange(max_seq_len, device=text_lengths.device)[None,:] < text_lengths[:,None]
            attn_mask = attn_mask.long() # Convert to 0-1

            _, text_embedding = self.bert(input_ids=input_text, attention_mask=attn_mask)
            text_logits = self.classifier(text_embedding)
            outputs['text_embed'], outputs['text_logits'] = text_embedding, text_logits
        return outputs

    def forward_text(self, input_text, text_lengths):
        outputs = {}
        batch_size = input_text.shape[0]
        max_seq_len = input_text.shape[1]
        # print(f"batch_size: {batch_size}, max_seq_len: {max_seq_len}")

        attn_mask = torch.arange(max_seq_len, device=text_lengths.device)[None,:] < text_lengths[:,None]
        attn_mask = attn_mask.long() # Convert to 0-1

        _, text_embedding = self.bert(input_ids=input_text, attention_mask=attn_mask)
        text_logits = self.classifier(text_embedding)

        # print(f"text_embedding: {text_embedding.size()}, text_logits: {text_logits.size()}")
        outputs['text_embed'], outputs['text_logits'] = text_embedding, text_logits
        return outputs

    # functions below are adopted from lugosch models.py
    def freeze_all_layers(self):
        for layer in self.lugosch_model.phoneme_layers:
            freeze_layer(layer)
        for layer in self.lugosch_model.word_layers:
            freeze_layer(layer)
    
    def print_frozen(self):
        for layer in self.lugosch_model.phoneme_layers:
            if has_params(layer):
                frozen = "frozen" if is_frozen(layer) else "unfrozen"
                print(layer.name + ": " + frozen)
        for layer in self.lugosch_model.word_layers:
            if has_params(layer):
                frozen = "frozen" if is_frozen(layer) else "unfrozen"
                print(layer.name + ": " + frozen)

    def unfreeze_one_layer(self):
        """
        ULMFiT-style unfreezing:
            Unfreeze the next trainable layer
        """
        # no unfreezing
        print("self.config.unfreezing_type: ", self.config.unfreezing_type)
        
        if self.config.unfreezing_type == 0:
            return

        if self.config.unfreezing_type == 1:
            trainable_index = 0 # which trainable layer
            global_index = 1 # which layer overall
            # print("Len of self.lugosch_model.word_layers:", len(self.lugosch_model.word_layers))
            while global_index <= len(self.lugosch_model.word_layers):
                layer = self.lugosch_model.word_layers[-global_index]
                # print("lugosch_model.word_layers[-global_index]:", layer)
                unfreeze_layer(layer)
                if has_params(layer): trainable_index += 1
                global_index += 1
                if trainable_index == self.unfreezing_index: 
                    self.unfreezing_index += 1
                    return

        if self.config.unfreezing_type == 2:
            trainable_index = 0 # which trainable layer
            global_index = 1 # which layer overall
            while global_index <= len(self.lugosch_model.word_layers):
                layer = self.lugosch_model.word_layers[-global_index]
                unfreeze_layer(layer)
                if has_params(layer): trainable_index += 1
                global_index += 1
                if trainable_index == self.unfreezing_index: 
                    self.unfreezing_index += 1
                    return

            global_index = 1
            while global_index <= len(self.lugosch_model.phoneme_layers):
                layer = self.lugosch_model.phoneme_layers[-global_index]
                unfreeze_layer(layer)
                if has_params(layer): trainable_index += 1
                global_index += 1
                if trainable_index == self.unfreezing_index:
                    self.unfreezing_index += 1
                    return

# codes from lugosch models.py
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True

def has_params(layer):
    num_params = sum([p.numel() for p in layer.parameters()])
    if num_params > 0: return True
    return False

def is_frozen(layer):
    for param in layer.parameters():
        if param.requires_grad: return False
    return True