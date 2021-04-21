from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import numpy as np
import torch.nn.functional as F
from data import get_dataloaders
from tqdm import tqdm
import os


def get_bert(pretrained=True, pretrained_model_name='bert-base-cased'):
    """Initialize a pretrained BERT model"""
    if pretrained:
        bert = BertModel.from_pretrained(pretrained_model_name)
    else:
        configuration = BertConfig.from_pretrained(pretrained_model_name)
        bert = BertModel(configuration)
    return bert

class BertNLU(nn.Module):
    """BERT NLU module"""
    def __init__(self, args, pretrained=True):
        super().__init__()
        self.bert = get_bert(pretrained)
        if args.dataset == "snips":
            num_classes = 6
        elif args.dataset == "fsc":
            num_classes = 31

        elif args.dataset == "slurp":
            num_classes = 91
            args.dataset = "snips"

        self.classifier = nn.Sequential(
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
    
    
class ExperimentRunner:
    def __init__(self, args):
        # Set the LR Scheduler and Loss Parameters
        self.args = args
        self.model = BertNLU(args)

        print(self.model)

        # Set the Device and Distributed Settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # Define the data loaders
        self.train_loader, \
        self.val_loader, \
        self.test_loader = get_dataloaders(data_root=args.data_path,
                                           batch_size=args.batch_size,
                                           dataset=args.dataset,
                                           num_workers=args.num_workers)
                                           

        # Define the optimizers
        if args.finetune_bert:
            print('Finetuning BERT')
            self.optimizer = torch.optim.Adam([
                {'params': self.model.bert.parameters(), 'lr':args.learning_rate_bert}, 
                {'params': self.model.classifier.parameters()}
            ], lr=args.learning_rate)
        else: 
            print('Freezing BERT')
            self.optimizer = torch.optim.Adam([ 
                {'params': self.model.classifier.parameters()}
            ], lr=args.learning_rate)

        
        if args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=0.5,
                                                                        patience=3,
                                                                        mode='max',
                                                                        verbose=True)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                 max_lr=args.learning_rate,
                                                                 steps_per_epoch=len(self.train_loader),
                                                                 epochs=args.num_epochs)
        self.criterion = torch.nn.CrossEntropyLoss()
      
        # Training specific params
        self.num_epochs = args.num_epochs
        self.print_every = args.print_every
        self.val_every = args.val_every
        self.model_dir = args.model_dir
        self.save_every = args.save_every

    def train(self):
        # Setting the variables before starting the training
        avg_train_loss = AverageMeter()
        avg_train_acc = AverageMeter()
        best_val_acc = -np.inf
    
        for epoch in range(self.num_epochs):

          
            avg_train_loss.reset()
            avg_train_acc.reset()

            # Mini batch loop
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                step = epoch * len(self.train_loader) + batch_idx

                # Get the model output for the batch and update the loss and accuracy meters
                train_loss, train_acc = self.train_step(batch)
                if self.args.scheduler == 'cycle':
                    self.scheduler.step()
                avg_train_loss.update([train_loss.item()])
                avg_train_acc.update([train_acc])
                
                # Logging and validation check
                if step % self.print_every == 0:
                    print('Epoch {}, batch {}, step {}, '
                          'loss = {:.4f}, acc = {:.4f}, '
                          'running averages: loss = {:.4f}, acc = {:.4f}'.format(epoch,
                                                                                 batch_idx,
                                                                                 step,
                                                                                 train_loss.item(),
                                                                                 train_acc,
                                                                                 avg_train_loss.get(),
                                                                                 avg_train_acc.get()))

                if step % self.val_every == 0:
                    val_loss, val_acc = self.val()
                    print('Val acc = {:.4f}, Val loss = {:.4f}'.format(val_acc, val_loss))
                   
                    # Update the save the best validation checkpoint if needed
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_chkpt_path = os.path.join(self.model_dir,
                                                       'best_ckpt.pth')
                        torch.save(self.model.state_dict(), best_chkpt_path)
                    if self.args.scheduler == 'plateau':
                        self.scheduler.step(val_acc)

    def compute_loss(self, batch):
        """ This function is specific to the kind of model we are training and must be implemented """
        

        batch['encoded_text'] = batch['encoded_text'].to(self.device)
        batch['text_length'] = batch['text_length'].to(self.device)
        batch['label'] = batch['label'].to(self.device)

        # Get the model outputs and the cross entropies
        output = self.model(batch['encoded_text'],
                            batch['text_length'])
        
        text_ce= self.criterion(output, batch['label'])
        
        predicted = torch.argmax(output, dim=1)
        correct = (predicted == batch['label'])
        accuracy = float(torch.sum(correct)) / predicted.shape[0]

        return {'loss': text_ce,
                'accuracy': accuracy,
                'predicted': predicted,
                'correct': correct,
                'model_output': output}


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        metrics = self.compute_loss(batch)
        metrics['loss'].backward()
        self.optimizer.step()
        return metrics['loss'], metrics['accuracy']

    def load_model_for_eval(self):
        chkpt_path = os.path.join(self.model_dir, 'best_ckpt.pth') \
            if self.args.eval_checkpoint_path is None else self.args.eval_checkpoint_path
        self.model.load_state_dict(torch.load(chkpt_path))
        self.model.eval()

    @torch.no_grad()
    def val(self):
        print('VALIDATING:')
        avg_val_loss = AverageMeter()
        avg_val_acc = AverageMeter()

        self.model.eval()
        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            metrics = self.compute_loss(batch)
            avg_val_acc.update(metrics['correct'].cpu().numpy())
            avg_val_loss.update([metrics['loss']])
        return avg_val_loss.get(), avg_val_acc.get()

    @torch.no_grad()
    def infer(self):
        self.load_model_for_eval()
        avg_test_loss = AverageMeter()
        avg_test_acc = AverageMeter()


        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            # Get the model output and update the meters
            output = self.compute_loss(batch) 
            avg_test_acc.update(output['correct'].cpu().numpy())
            avg_test_loss.update([output['loss']])

        print('Final test acc = {:.4f}, test loss = {:.4f}'.format(avg_test_acc.get(), avg_test_loss.get()))
        return avg_test_loss.get(), avg_test_acc.get()
    
    
    
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_val = 0.0
        self.count = 0

    def update(self, values):
        self.sum_val += np.sum(values)
        self.count += len(values)

    def get(self):
        return self.sum_val / self.count
