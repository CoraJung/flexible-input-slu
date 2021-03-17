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

import argparse

def parse():
    parser_impatient = argparse.Argumentparser_impatient(description="Run an experiment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_impatient.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Seed for the RNG")

    parser_impatient.add_argument("--dataset",
                        choices=['fsc', 'snips', 'slurp'],
                        default='fsc',
                        help="The dataset to use")
    
    # parser_impatient.add_argument("--num-classes",
    #                     type=int,
    #                     help="Number of classes that a dataset predicts")

    parser_impatient.add_argument("--data-path",
                        type=str,
                        help="Path to the data folder containing data csvs")

    parser_impatient.add_argument("--experiment",
                        default="experiments.experiment_triplet.ExperimentRunnerTriplet",
                        help="Experiment to run")

    parser_impatient.add_argument("-lr", "--learning-rate",
                        type=float,
                        default=1e-3,
                        help="Learning rate of the acoustic encoder")

    parser_impatient.add_argument("-lr-bert", "--learning-rate-bert",
                        type=float,
                        default=2e-5,
                        help="Learning rate of the BERT model")

    parser_impatient.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Batch size")

    parser_impatient.add_argument("--num-epochs",
                        type=int,
                        default=20,
                        help="Number of epochs to the train the model`")

    parser_impatient.add_argument("--print-every",
                        type=int,
                        default=20,
                        help="Print model stats every n steps")

    parser_impatient.add_argument("--val-every",
                        type=int,
                        default=50,
                        help="Validate model every n steps`")

    parser_impatient.add_argument("--save-every",
                        type=int,
                        default=500,
                        help="Save model every n steps`")

    parser_impatient.add_argument("--infer-only",
                        action='store_true',
                        help="Only run inference on the saved model")

    parser_impatient.add_argument("--visualize",
                        action='store_true',
                        help="User tensorboard for visualizing training curves")

    parser_impatient.add_argument("--distributed",
                        action='store_true',
                        help="Use multiple GPUs for training")

    parser_impatient.add_argument("--bert-random-init",
                        action='store_true', #default value of false
                        help="Use a randomly initialized BERT model")

    parser_impatient.add_argument("--num-workers",
                        type=int,
                        default=7,
                        help="Number of concurrent dataloader threads")

    parser_impatient.add_argument("-m", "--margin",
                        type=float,
                        default=1.0,
                        help="Margin for embedding losses")

    parser_impatient.add_argument("--g-steps",
                        type=int,
                        default=1,
                        help="Number of generator steps per batch")

    parser_impatient.add_argument("--d-steps",
                        type=int,
                        default=5,
                        help="Number of discriminator steps per batch")

    parser_impatient.add_argument("--scheduler",
                        choices=['plateau', 'cycle', 'none'],
                        default='none',
                        help="Learning rate scheduler")

    parser_impatient.add_argument("--bert-model-name",
                        default="bert-base-cased",
                        help="Name or path of pretrained BERT model to use")

    parser_impatient.add_argument("--num-enc-layers",
                        default=3,
                        type=int,
                        help="Number of encoder LSTM layers")

    parser_impatient.add_argument("--enc-dim",
                        default=256,#original 512
                        type=int,
                        help="Hidden dimension of encoder LSTM")

    parser_impatient.add_argument("--weight-audio",
                        default=1,
                        type=float,
                        help="Weight of the audio classification loss for joint models")

    parser_impatient.add_argument("--weight-text",
                        default=1,
                        type=float,
                        help="Weight of the text classification loss for joint models")

    parser_impatient.add_argument("--weight-embedding",
                        default=1,
                        type=float,
                        help="Weight of the embedding loss for joint models")

    parser_impatient.add_argument("--weight-adversarial",
                        default=1,
                        type=float,
                        help="Weight of the adversarial loss for joint models")

    parser_impatient.add_argument("--model-dir",
                        default='./',
                        type=str,
                        help="Directory to store the trained model checkpoints.")

    parser_impatient.add_argument("--eval-checkpoint-path",
                        default=None,
                        type=str,
                        help="Checkpoint path to be used for testing.")

    ### Lugosch's Pretrained Model parser_impatient ###
    parser_impatient.add_argument("--use-sincnet",
                        default=True)

    parser_impatient.add_argument("--fs",
                        default=16000)

    parser_impatient.add_argument("--cnn-N-filt",
                        nargs="+",
                        default=[80,60,60])

    parser_impatient.add_argument("--cnn-len-filt",
                        nargs="+",
                        default=[401,5,5])

    parser_impatient.add_argument("--cnn-stride",
                        nargs="+",
                        default=[80,1,1])

    parser_impatient.add_argument("--cnn-max-pool-len",
                        nargs="+",
                        default=[2,1,1])

    parser_impatient.add_argument("--cnn-act",
                        nargs="+",
                        default=["leaky_relu","leaky_relu","leaky_relu"])

    parser_impatient.add_argument("--cnn-drop",
                        nargs="+",
                        default=[0.0,0.0,0.0])

    parser_impatient.add_argument("--phone-rnn-num-hidden",
                        nargs="+",
                        default=[128,128])

    parser_impatient.add_argument("--phone-downsample-len",
                        nargs="+",
                        default=[2,2])

    parser_impatient.add_argument("--phone-downsample-type",
                        nargs="+",
                        default=["avg","avg"])

    parser_impatient.add_argument("--phone-rnn-drop",
                        nargs="+",
                        default=[0.5,0.5])

    parser_impatient.add_argument("--phone-rnn-bidirectional",
                        default=True)

    parser_impatient.add_argument("--word-rnn-num-hidden",
                        nargs="+",
                        default=[128,128])

    parser_impatient.add_argument("--word-downsample-len",
                        nargs="+",
                        default=[2,2])

    parser_impatient.add_argument("--word-downsample-type",
                        nargs="+",
                        default=["avg","avg"])

    parser_impatient.add_argument("--word-rnn-drop",
                        nargs="+",
                        default=[0.5,0.5])

    parser_impatient.add_argument("--word-rnn-bidirectional",
                        default=True)

    parser_impatient.add_argument("--vocabulary-size",
                        default=10000)

    parser_impatient.add_argument("--libri-folder",
                        default="/misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/config")

    parser_impatient.add_argument("--unfreezing-type",
                        type=int,
                        default=2,
                        help="0: No Unfreezing (freeze all), 1: Unfreeze Word, 2: Unfreeze All")

    ### Lugosch's IntentModule parser_impatient ###
    parser_impatient.add_argument("--intent-rnn-num-hidden",
                        default=[128],
                        nargs="+")
    parser_impatient.add_argument("--intent-downsample-len",
                        default=[1],
                        nargs="+")
    parser_impatient.add_argument("--intent-downsample-type",
                        default=["none"],
                        nargs="+")
    parser_impatient.add_argument("--intent-rnn-drop",
                        default=[0.5],
                        nargs="+")
    parser_impatient.add_argument("--intent-rnn-bidirectional",
                        action="store_false") #True
    
    ### ASR parser_impatient ###
    parser_impatient.add_argument("-lr-bert-asr", "--learning-rate-bert-asr",
                        type=float,
                        default=2e-5,
                        help="Learning rate of the ASR BERT model")

    parser_impatient.add_argument("--weight-asr",
                        default=1,
                        type=float,
                        help="Weight of the ASR text classification loss for joint models")

    parser_impatient.add_argument("-am", "--asr-margin",
                        type=float,
                        default=1.0,
                        help="Margin for ASR embedding losses")

    parser_impatient.add_argument("--weight-embedding-asr",
                        default=1,
                        type=float,
                        help="Weight of the ASR embedding loss for joint models")
    
    ### bert parser_impatient ###
    parser_impatient.add_argument("--finetune-bert",
                        action='store_true',
                        help="decides if bert model is finetuned or frozen. default is false")

    ### bert loader parser_impatient ###
    parser_impatient.add_argument("--bert-dir",
                        default=None,
                        help="load finetuned bert model")
    
    ### save criteria ###
    parser_impatient.add_argument("--model-save-criteria",
                        choices=['combined', 'audio_text'],
                        default='audio_text',
                        help="Criteria to save the best checkpoints")
    
    ### max_patience ###
    parser_impatient.add_argument("--max-patience",
                        type=int,
                        default=5,
                        help="Max patience for early stopping")
    
    ### checkpoint path ###
    parser_impatient.add_argument("--checkpoint-dir",
                        default=None,
                        help="load best model state")

    args = parser_impatient.parse_args()
    return args
