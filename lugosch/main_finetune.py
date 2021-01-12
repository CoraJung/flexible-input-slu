import torch
import numpy as np
from models import PretrainedModel, Model
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training_finetune import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='run ASR pre-training')
parser.add_argument('--train', action='store_true', help='run SLU training')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters')

###---Emmy edited on 10/29/2020---###
### Add new arg to pass in the model dir for code pretrained on fluent.ai ###
parser.add_argument('--model_path', type=str, help='path to folder containing the pretrained model')
parser.add_argument('--max_patience', type=int, default = 5, help='max patience for early stopping')
parser.add_argument('--training_lr', type=float, default = 0.001, help='set training learning rate')

###---Cora edited on 11/22/2020---###
parser.add_argument('--model_state_num', type=int, default = 42, help='load model state dict')


args = parser.parse_args()
pretrain = args.pretrain
train = args.train
restart = args.restart
config_path = args.config_path
# 10/29/2020
model_path = args.model_path
max_patience = args.max_patience
training_lr = args.training_lr
# 11/22/2020
model_state_num = args.model_state_num

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

if pretrain:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

	# Initialize base model
	pretrained_model = PretrainedModel(config=config)

	# Train the base model
	trainer = Trainer(model=pretrained_model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.pretraining_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
		train_phone_acc, train_phone_loss, train_word_acc, train_word_loss = trainer.train(train_dataset)
		valid_phone_acc, valid_phone_loss, valid_word_acc, valid_word_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
		print("*phonemes*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_phone_acc, train_phone_loss, valid_phone_acc, valid_phone_loss) )
		print("*words*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_word_acc, train_word_loss, valid_word_acc, valid_word_loss) )

		trainer.save_checkpoint()

if train:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)

	# Initialize final model
	model = Model(config=config)

	# Train the final model
	trainer = Trainer(model=model, config=config, lr = training_lr) #add training_lr as hyper param 11/08 Em.
    
###---Emmy edited on 10/29/2020---###
### Add new arg to pass in the model dir for code pretrained on fluent.ai ###
	if restart: 
		print("loading_checkpoint")
		trainer.load_checkpoint(model_path= model_path, model_state_num=model_state_num)
	
	val_accuracies=[]
	best_valid_intent_acc = 0
	patience_counter = 0
	#max_patience = 5 #added as a hyper param above
        
	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.train(train_dataset)
		valid_intent_acc, valid_intent_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )

		############################### cora edit 11/5 ###############################
		val_accuracies.append(valid_intent_acc)
		if val_accuracies[-1] > best_valid_intent_acc:
			best_valid_intent_acc = val_accuracies[-1]
			patience_counter = 0
			trainer.save_checkpoint()
                
		else:
			patience_counter += 1    
			if patience_counter > max_patience:
				break
	
	model_name = config.folder.split("/")[-1]
	print("best validation intent accuracy is: ", best_valid_intent_acc)
	print("the training learning rate is: ", training_lr, " unfreezing_type: ", config.unfreezing_type, " the model: ", model_name)

	##################### Edit by Wendy
	test_intent_acc, test_intent_loss = trainer.test(test_dataset)
	print("========= Test results =========")
	print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )
