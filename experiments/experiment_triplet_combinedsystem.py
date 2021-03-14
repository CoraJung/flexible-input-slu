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
from models.model_combined import JointModel
from dataloader.data import get_triplet_dataloaders
from experiments.experiment_base_combinedsystem import ExperimentRunnerBase
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
        
        elif args.dataset == 'slurp':
            num_classes = 91
            args.dataset = 'snips'
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
        if args.finetune_bert:
            print('Finetuning BERT')
            self.optimizer = torch.optim.Adam([
                {'params': self.model.bert.parameters(), 'lr':args.learning_rate_bert}, 
                {'params': self.model.lugosch_model.parameters()},
                {'params': self.model.classifier.parameters()}
            ], lr=args.learning_rate)
        else: 
            print('Freezing BERT')
            self.optimizer = torch.optim.Adam([ 
                {'params': self.model.lugosch_model.parameters()},
                {'params': self.model.classifier.parameters()}
            ], lr=args.learning_rate)
        
        # Parameters for the losses
        self.weight_text = args.weight_text
        self.weight_embedding = args.weight_embedding
        self.margin = args.margin

        super().__init__(args)

    def compute_loss(self, batch):
        print('computing loss...')
        batch['feats'] = batch['feats'].to(self.device)
        batch['length'] = batch['length'].to(self.device)
        batch['encoded_text'] = batch['encoded_text'].to(self.device)
        batch['text_length'] = batch['text_length'].to(self.device)
        batch['label'] = batch['label'].to(self.device)

        # Get the model outputs and the cross entropies
        print('getting model outputs...')
        output = self.model(batch['feats'],
                            batch['length'],
                            batch['encoded_text'],
                            batch['text_length'])
        print('calc cross entropy...')
        audio_ce = self.criterion(output['audio_logits'], batch['label'])
        print('audio logits:', output['audio_logits'], ', label:', batch['label'])
        text_ce = self.criterion(output['text_logits'], batch['label'])
        print('text logits:', output['text_logits'], ', label:', batch['label'])

        # Triplet loss - positive instance
        print('getting positive instances for triplet loss...')
        batch['encoded_text2'] = batch['encoded_text2'].to(self.device)
        batch['text_length2'] = batch['text_length2'].to(self.device)
        with torch.no_grad():
            output_pos = self.model(input_text=batch['encoded_text2'],
                                    text_lengths=batch['text_length2'],
                                    text_only=True)
        
        print('getting negative instances...')
        # Triplet loss - negative instance
        batch['encoded_text3'] = batch['encoded_text3'].to(self.device)
        batch['text_length3'] = batch['text_length3'].to(self.device)
        with torch.no_grad():
            output_neg = self.model(input_text=batch['encoded_text3'],
                                    text_lengths=batch['text_length3'],
                                    text_only=True)

        embed_dist_pos = ((output['audio_embed']-output_pos['text_embed'].detach())**2).mean(-1)
        embed_dist_neg = ((output['audio_embed']-output_neg['text_embed'].detach())**2).mean(-1)
        triplet_loss = F.relu(self.margin + embed_dist_pos - embed_dist_neg)
        triplet_loss = triplet_loss.mean()

        # Define the joint loss
        print('calc joint loss...')
        loss = audio_ce + \
               (self.weight_text * text_ce) + \
               (self.weight_embedding * triplet_loss)

        predicted = torch.argmax(output['audio_logits'], dim=1)
        correct = (predicted == batch['label'])
        accuracy = float(torch.sum(correct)) / predicted.shape[0]
        
        #Add combined system
        print('system combination...')
        combined_logits = (output['audio_logits']+output['text_logits'])/2
        combined_predicted = torch.argmax(combined_logits, dim=1)
        combined_correct = (combined_predicted == batch['label'])
        combined_accuracy = float(torch.sum(combined_correct)) / combined_predicted.shape[0]

        # Accuracy of text branch
        print('calc text branch acc...')
        text_predicted = torch.argmax(output['text_logits'], dim=1)
        text_correct = (text_predicted == batch['label'])
        text_accuracy = float(torch.sum(text_correct)) / text_predicted.shape[0]

        return {'loss': loss,
                'accuracy': accuracy,
                'predicted': predicted,
                'correct': correct,
                'model_output': output,
                'text_accuracy': text_accuracy,
                'text_predicted': text_predicted,
                'text_correct': text_correct,
                'combined_accuracy': combined_accuracy,
                'combined_predicted': combined_predicted,
                'combined_correct': combined_correct}
