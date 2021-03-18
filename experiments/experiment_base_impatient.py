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
import os

import torch
import numpy as np
from utils.utils import AverageMeter
from tqdm import tqdm
# from utils.visualize import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


class ExperimentRunnerBase:
    def __init__(self, args):
        # Set the LR Scheduler and Loss Parameters
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
        self.args = args
        self.num_epochs = args.num_epochs
        self.print_every = args.print_every
        self.val_every = args.val_every
        self.model_dir = args.model_dir
        self.save_every = args.save_every
        self.max_patience = args.max_patience
        print('Max_patience: ', self.max_patience)

    def train(self):
        # Setting the variables before starting the training
        print('Loading checkpoint if checkpoint_dir is given...')
        self.load_checkpoint()
        
        avg_train_loss = AverageMeter()
        avg_train_acc = AverageMeter()
        text_avg_train_acc = AverageMeter()
        combined_avg_train_acc = AverageMeter()
        
        best_val_acc = -np.inf
        patience_counter = 0
        best_epoch = self.num_epochs
        for epoch in range(self.num_epochs):

            self.model.print_frozen()

            avg_train_loss.reset()
            avg_train_acc.reset()
            text_avg_train_acc.reset()
            

            # Mini batch loop
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                step = epoch * len(self.train_loader) + batch_idx

                # Get the model output for the batch and update the loss and accuracy meters
                train_loss, train_acc, text_train_acc = self.train_step(batch)
                if self.args.scheduler == 'cycle':
                    self.scheduler.step()
                avg_train_loss.update([train_loss.item()])
                avg_train_acc.update([train_acc])
                text_avg_train_acc.update([text_train_acc])
                
                # Logging and validation check
                if step % self.print_every == 0:
                    print('Epoch {}, batch {}, step {}, '
                          'loss = {:.4f}, acc_audio = {:.4f}, acc_text = {:.4f}, '
                          'running averages: loss = {:.4f}, acc_audio = {:.4f}, acc_text = {:.4f}'.format(epoch,
                                                                                 batch_idx,
                                                                                 step,
                                                                                 train_loss.item(),
                                                                                 train_acc,
                                                                                 text_train_acc,
                                                                                 avg_train_loss.get(),
                                                                                 avg_train_acc.get(),
                                                                                 text_avg_train_acc.get()))

                if step % self.val_every == 0:
                    val_loss, val_acc, text_val_acc, combined_val_acc = self.val()
                    print('Val acc (audio) = {:.4f}, Val acc (text) = {:.4f}, Val acc (combined) = {:.4f}, Val loss = {:.4f}'.format(val_acc, text_val_acc, combined_val_acc, val_loss))

                    # Update the save the best validation checkpoint if needed
                    if self.args.model_save_criteria == 'audio_text':
                        cur_avg_acc = (val_acc + text_val_acc) / 2
                    else: #'combined'
                        cur_avg_acc = combined_val_acc
                    
                    if cur_avg_acc > best_val_acc:
                        #print('Start saving best check point at step{}...'.format(step))
                        best_val_acc = cur_avg_acc
                        best_chkpt_path = os.path.join(self.model_dir,
                                                       'best_ckpt.pth')
                        torch.save(self.model.state_dict(), best_chkpt_path)
                        print('Done saving best check point!')
                    if self.args.scheduler == 'plateau':
                        self.scheduler.step(audio_text_avg_acc)

            print('------ End of epoch validation ------')
            val_loss, val_acc, text_val_acc, combined_val_acc = self.val()
            # Update the save the best validation checkpoint if needed
            if self.args.model_save_criteria == 'audio_text':
                cur_avg_acc = (val_acc + text_val_acc) / 2
            else: #'combined'
                cur_avg_acc = combined_val_acc
            
            if cur_avg_acc > best_val_acc:
                #print('Start saving best check point at step{}...'.format(step))
                best_val_acc = cur_avg_acc
                best_chkpt_path = os.path.join(self.model_dir,
                                                'best_ckpt.pth')
                torch.save(self.model.state_dict(), best_chkpt_path)
                patience_counter = 0
                best_epoch = epoch
                print('Done saving best check point! Patience counter reset!')
            else:
                patience_counter += 1    
                if patience_counter > self.max_patience:
                    print('Reach max patience limit. Training stops! Best val acc achieved at epoch: {}.'.format(epoch))
                    break
            self.model.unfreeze_one_layer()


    def compute_loss(self, batch):
        """ This function is specific to the kind of model we are training and must be implemented """
        raise NotImplementedError

    def train_step(self, batch):
        
        self.model.train()
        self.optimizer.zero_grad()
        metrics = self.compute_loss(batch)
        metrics['loss'].backward()
        self.optimizer.step()
        return metrics['loss'], metrics['accuracy'], metrics['text_accuracy']

    def load_model_for_eval(self):
        chkpt_path = os.path.join(self.model_dir, 'best_ckpt.pth') \
            if self.args.eval_checkpoint_path is None else self.args.eval_checkpoint_path
        self.model.load_state_dict(torch.load(chkpt_path))
        self.model.eval()
    
    def load_checkpoint(self):
        if self.args.checkpoint_dir:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, 'best_ckpt.pth') 
            print(f"Checkpoint path is given as {checkpoint_path}")         
            if os.path.isfile(checkpoint_path):
                print("Found best_ckpt.pth in given model path.")
                try:
                    self.model.load_state_dict(torch.load(checkpoint_path))
                    print("Successfully loaded best checkpoint")
                    
                except:
                    print("Could not load previous model; starting from scratch")
        else:
            print("No previous model; starting from scratch")
        self.model.train()

    @torch.no_grad()
    def val(self):
        print('VALIDATING:')
        avg_val_loss = AverageMeter() #final loss
        avg_val_acc = AverageMeter()  #audio acc
        text_avg_val_acc = AverageMeter() #text acc
        combined_avg_val_acc = AverageMeter() #combined acc
        
        self.model.eval()
        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            
            metrics = self.compute_loss(batch)
            avg_val_acc.update(metrics['correct'].cpu().numpy())
            text_avg_val_acc.update(metrics['text_correct'].cpu().numpy())
            combined_avg_val_acc.update(metrics['combined_correct'].cpu().numpy())
            
            avg_val_loss.update([metrics['loss']])
        return avg_val_loss.get(), avg_val_acc.get(), text_avg_val_acc.get(), combined_avg_val_acc.get()

    @torch.no_grad()
    def infer(self):
        
        self.load_model_for_eval()
        avg_test_loss = AverageMeter()
        avg_test_acc = AverageMeter()
        text_avg_test_acc = AverageMeter()
        combined_avg_test_acc = AverageMeter()

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            # Get the model output and update the meters
            output = self.compute_loss(batch)
            avg_test_acc.update(output['correct'].cpu().numpy())
            text_avg_test_acc.update(output['text_correct'].cpu().numpy())
            combined_avg_test_acc.update(output['combined_correct'].cpu().numpy())
            
            avg_test_loss.update([output['loss']])

        print('Final test acc (audio) = {:.4f}, final test acc (text) = {:.4f}, final test acc (combined system) = {:.4f}, test loss = {:.4f}'.format(avg_test_acc.get(), text_avg_test_acc.get(), combined_avg_test_acc.get(), avg_test_loss.get()))
        return avg_test_loss.get(), avg_test_acc.get(), text_avg_test_acc.get()
