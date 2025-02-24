<p align="center">
  <img src="https://github.com/user-attachments/assets/482e5ed5-e5d2-47d1-8da2-68211cda9c74" />
</p>

# Project Echo CNN
## Description
This project aims to develop a **Noise Cancelation System** to be used to remove background noise from the input audio by using Convolutional Neural Network's in Pytorch. This model transforms .wav files into mel spectrograms feeds it into the CNN produces a clean spectrogram and converts it back into waveform and stores it into a .wav file. 
## Table of Contents
- [Description](#description)
- [Table of Contents](#table-of-contents)
- [Technologies Used](#technologies-used)
- [Requirements](#requirements)
  - [Required](#required)
  - [Optional](#optional)
- [Usage Instructions](#usage-instructions)
  - [Training](#training-model)
    - [Example](#examples)
  - [Loading](#loading-model)
- [Project Roadmap](#project-roadmap)
## Technologies Used
- Python
- Pytorch
- Torchaudio (set to change)
## Requirements

### Required
- Python 3.13+
- Python packages:
  - librosa==0.10.2.post1
  - matplotlib==3.10.0
  - numpy==2.2.3
  - setuptools==75.8.0
  - torch==2.6.0+cu126
  - torchaudio==2.6.0+cu126
---
### Optional
- Dataset: [Kaggle Dataset](url)
## Usage Instructions
### Training Model

To train a model with the architecture in the ``src.models.model`` Module:
- Ensure that the directory of **data** is created where the clean, noisy data for training and testing will be the directories are specified in the ``__init__.py`` file.
```python
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "output")
MODEL_OUPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "output.wav")

DATA_DIR = os.path.join(ROOT_DIR, "data")

NOISY_TRAIN_DIR = os.path.join(DATA_DIR, "noisy_trainset_wav")
CLEAN_TRAIN_DIR = os.path.join(DATA_DIR, "clean_trainset_wav")

NOISY_TEST_DIR = os.path.join(DATA_DIR, "noisy_testset_wav")
CLEAN_TEST_DIR = os.path.join(DATA_DIR, "clean_testset_wav")
```
 - Load the data using the data loader in the ``utils.data_loader`` module
 - Instantiate new instance of the model
 - Call the train function and enter the relevant parameters

i.e:
```python
model = CNN().to(device)

spectrogram = MelSpectrogram(n_fft=1024)

data_loader = load_data(CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR, spectrogram)

updated_state = train(
    model=model,
    epochs=120,
    optimiser=Adam(model.parameters()),
    loss_func=torch.nn.MSELoss(),
    data_loader=data_loader
)
```
- Save the updated state to a model.pt file
#### Examples

---
### Loading Model
```python
load_model = src.utils.load_model()
model = src.models.model.CNN().to(device)
model.load_state_dict(load_model)
model.eval()
```
---

## Project Roadmap
