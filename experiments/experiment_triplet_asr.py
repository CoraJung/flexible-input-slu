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

import torch
from models.model_combined_asr import JointModel
from dataloader.data_asr import get_triplet_dataloaders
from experiments.experiment_base_asr import ExperimentRunnerBase
import torch.nn.functional as F


"""Runner class which implements training with triplet loss"""
class ExperimentRunnerTriplet(ExperimentRunnerBase):
    def __init__(self, args):

        # Get the dataset directory
        if args.data_path:
            data_dir = args.data_path
        else:
            raise ValueError("No data path was given!")

        # # Get the correct dataset directory
        if args.dataset == 'fsc':
            # data_dir = 'fluent'
            num_classes = 31
        elif args.dataset == 'snips':
            # data_dir = 'snips_slu'
            num_classes = 6
        else:
            raise ValueError("No valid dataset selected!")

        print(f"the dataset we are using is: {args.dataset}")

        # Define the joint model
        self.model = JointModel(input_dim=40,
                                num_layers=args.num_enc_layers,
                                num_classes=num_classes,
                                encoder_dim=args.enc_dim,#128
                                bert_pretrained=not args.bert_random_init, # == True (not False) in default, true
                                bert_pretrained_model_name=args.bert_model_name,
                                config=args)

        print(self.model)

        # Set the Device and Distributed Settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # Define the data loaders
        self.train_loader, \
        self.val_loader, \
        self.test_loader = get_triplet_dataloaders(data_root=data_dir,
                                                   batch_size=args.batch_size,
                                                   dataset=args.dataset,
                                                   num_workers=args.num_workers,
                                                   pretrained_model_name=args.bert_model_name)

        # Define the optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.model.bert.parameters(), 'lr':args.learning_rate_bert},
            {'params': self.model.bert_asr.parameters(), 'lr':args.learning_rate_bert_asr},
            {'params': self.model.lugosch_model.parameters()}, # replace speech_encoder with lugosch_model
            {'params': self.model.classifier.parameters()}
        ], lr=args.learning_rate)

        
        # Parameters for the losses
        self.weight_audio = args.weight_audio
        self.weight_text = args.weight_text
        self.weight_embedding = args.weight_embedding #audio
        self.weight_embedding_asr = args.weight_embedding_asr
        self.weight_asr = args.weight_asr
        self.margin = args.margin

        super().__init__(args)

    def compute_loss(self, batch):
        batch['feats'] = batch['feats'].to(self.device)
        batch['length'] = batch['length'].to(self.device)
        batch['encoded_text'] = batch['encoded_text'].to(self.device)
        batch['text_length'] = batch['text_length'].to(self.device)
        
        batch['encoded_asr'] = batch['encoded_asr'].to(self.device)
        batch['asr_length'] = batch['asr_length'].to(self.device)
        batch['label'] = batch['label'].to(self.device)

        # Get the model outputs and the cross entropies
        output = self.model(batch['feats'],
                            batch['length'],
                            batch['encoded_text'],
                            batch['encoded_asr'],
                            batch['text_length'],
                            batch['asr_length'])
        
        audio_ce = self.criterion(output['audio_logits'], batch['label'])
        text_ce = self.criterion(output['text_logits'], batch['label'])
        asr_ce = self.criterion(output['asr_logits'], batch['label'])

        # Triplet loss - positive instance
        batch['encoded_text2'] = batch['encoded_text2'].to(self.device)
        batch['text_length2'] = batch['text_length2'].to(self.device)
        
        with torch.no_grad():
            output_pos = self.model(input_text=batch['encoded_text2'],
                                    text_lengths=batch['text_length2'],
                                    text_only=True)

        # Triplet loss - negative instance
        batch['encoded_text3'] = batch['encoded_text3'].to(self.device)
        batch['text_length3'] = batch['text_length3'].to(self.device)
        with torch.no_grad():
            output_neg = self.model(input_text=batch['encoded_text3'],
                                    text_lengths=batch['text_length3'],
                                    text_only=True)
        # Compare audio embeddings with text embeddings of pos/neg samples
        embed_dist_pos = ((output['audio_embed']-output_pos['text_embed'].detach())**2).mean(-1)
        embed_dist_neg = ((output['audio_embed']-output_neg['text_embed'].detach())**2).mean(-1)
        triplet_loss = F.relu(self.margin + embed_dist_pos - embed_dist_neg)
        triplet_loss = triplet_loss.mean()
        
        # Compare asr embeddings with text embeddings of pos/neg samples
        asr_embed_dist_pos = ((output['asr_embed']-output_pos['text_embed'].detach())**2).mean(-1)
        asr_embed_dist_neg = ((output['asr_embed']-output_neg['text_embed'].detach())**2).mean(-1)
        asr_triplet_loss = F.relu(self.asr_margin + asr_embed_dist_pos - asr_embed_dist_neg)
        asr_triplet_loss = asr_triplet_loss.mean()

        # Define the joint loss
        loss = (self.weight_audio * audio_ce) + \
               (self.weight_asr * asr_ce) + \
               (self.weight_text * text_ce) + \
               (self.weight_embedding * triplet_loss) +\
               (self.weight_embedding_asr * asr_triplet_loss)
        # Accuracy of audio branch
        predicted = torch.argmax(output['audio_logits'], dim=1)
        correct = (predicted == batch['label'])
        accuracy = float(torch.sum(correct)) / predicted.shape[0]
         
        # Accuracy of asr branch
        asr_predicted = torch.argmax(output['asr_logits'], dim=1)
        asr_correct = (asr_predicted == batch['label'])
        asr_accuracy = float(torch.sum(asr_correct)) / asr_predicted.shape[0]
        
       
        return {'loss': loss,
                'accuracy': accuracy,
                'predicted': predicted,
                'correct': correct,
                'model_output': output,
                'asr_accuracy': asr_accuracy,
                'asr_predicted': asr_predicted,
                'asr_correct': asr_correct}
