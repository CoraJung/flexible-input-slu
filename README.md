# IBM-NYU Project
## ASR-Text-Speech End-to-End SLU Model

The baseline code is adopted from Alexa End-to-End SLU System: https://github.com/alexa/alexa-end-to-end-slu. 
The original setup is a cross-modal system that co-trains text embeddings and acoustic embeddings in a shared latent space.
We further enhance this system by utilizing an acoustic module pre-trained on LibriSpeech and domain-adapting the text module on our target datasets.

The model uses the Snips SLU, the Fluent Speech dataset (FSC), or SLURP SLU with speech, ground truth and ASR transcripts as training inputs.
This framework is built using pytorch with torchaudio and the transformer package from HuggingFace.
We tested using pytorch 1.5.0 and torchaudio 0.5.0.

## Installation and data preparation

The environment set up is the same as alexa-slu git repository.
To install the required python packages, please run `pip install -r requirements.txt`. This setup uses the `bert-base-cased` model.
Typically, the model will be downloaded (and cached) automatically when running the training for the first time.
In case you want to download the model explicitly, you can run the `download_bert.py` script from the `dataprep/` directory,
e.g. `python download_bert.py bert-base-cased ./models/bert-base-cased`. 

To preprocess the Snips dataset, please run `prepare_snips.py` (located in the `dataprep/` directory) from within the `snips_slu/` folder dataset.
This will generate additional files within the `snips_slu/` folder required by the dataloader.

## Running experiments

Core to running experiments is the `train.py` script.
When called without any parameters, it will train a model using triplet loss on the FSC dataset.
The default location for saving intermediate results is the pre-specified `--model-dir` directory.
In case it does not yet exist, it will be created.

To customize the experiments, several command line options are available (for a full list, please refer to `parser.py`):

* --dataset (The dataset to use, e.g. `fsc`)
* --experiment (The experiment class to run, e.g. `experiments.experiment_triplet.ExperimentRunnerTriplet`)
* --infer-only (Only run inference on the saved model)
* --learning-rate-bert (The learning rate for BERT branch only)
* --learning-rate (The learning for the acoustic branch and shared classifier)
* --scheduler (Learning rate scheduler)
* --finetune-bert (A boolean parameter indicating whether or not to fine-tune pre-trained BERT)
* --bert-dir (The directory to load pre-trained or domain-adapted BERT model)
* --model-save-criteria (The criteria to select the best checkpoints, e.g. best validation audio accuracy or the average of validation audio + text accuracy)
* --model-dir (The directory to save the best checkpoint)

## Example runs

To check if everything is installed correctly, training a model with either Snips SLU or Fluent Speech Commands should produce the following results:

### Fluent Speech Commands

`python train.py --dataset fsc`

Final test acc = 0.9565, test loss = 0.5085

### Snips SLU

`python train.py --dataset snips`

Final test acc = 0.6988, test loss = 2.2471


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
