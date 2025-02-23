import torch
import torchaudio
import matplotlib.pyplot as plt 
import utils
import numpy as np
from Data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(149504, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128 * 586),
        )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.nn(x)
        x = x.view(-1, 128, 586)
        return x
# Plot FFT magnitude spectrum
model = CNN().to(device=device)

spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=1024)
train_noisy_dir = "data/noisy_trainset_wav"
train_clean_dir = "data/clean_trainset_wav"
test_noisy_dir = "data/noisy_testset_wav"
test_clean_dir = "data/clean_testset_wav"

train_set = Data(train_noisy_dir, train_clean_dir, spectrogram)
test_set = Data(test_noisy_dir, test_clean_dir, spectrogram)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=lambda b: utils.collate_fn(b, target_time=586))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=lambda b: utils.collate_fn(b, target_time=586))
print("Finished Loading Data")
# ------------------ Training Code ------------------
# Set training parameters
num_epochs = 120 
learning_rate = 0.05

# Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)



# After training, display the final output spectrogram
model.eval()
model.to("cpu")
torch.save(model.state_dict(),'model_scripted.pt')
# final_output = model(input_tensor).squeeze()  # Remove the batch dimension
# print(final_output.size())
# inv_mel = torchaudio.transforms.InverseMelScale(513)
# utils.plot_spectrogram(final_output.detach(), "Trained Output")
# grif = torchaudio.transforms.GriffinLim(n_fft=1024)
# final_output = grif(inv_mel(final_output.unsqueeze(0)))
# print(final_output.size())
# torchaudio.save(".\\wavFiles\\output.wav", final_output.detach(), noisy_sample_rate)
# plt.show()
