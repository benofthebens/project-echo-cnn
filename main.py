from src.models.model import CNN
from src.utils.save_model import save_model
from src.models.train import train
from src.utils.load_data import load_data
from src import CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR
from torch import cuda
from torch.optim import Adam
from torchaudio.transforms import MelSpectrogram
import torch

# Will use the GPU if it is available
device = torch.device("cuda" if cuda.is_available() else "cpu")

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

save_model(updated_state)


