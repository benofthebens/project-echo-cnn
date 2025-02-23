import torch
import torchaudio
import os
class Data(torch.utils.data.Dataset):
    def __init__(self, noise_dir, clean_dir, transform=None):
        super().__init__()
        self.noise_dir = noise_dir
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(noise_dir))
        self.transform = transform
    def __len__(self):
        return len(self.files) 

    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        # Load audio
        noisy_path = os.path.join(self.noise_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        noisy_waveform, sr = torchaudio.load(noisy_path)
        clean_waveform, _ = torchaudio.load(clean_path)  # Same sample rate expected

        # Apply transformations if provided
        if self.transform != None:
            noisy_waveform = self.transform(noisy_waveform)
            clean_waveform = self.transform(clean_waveform)

        return noisy_waveform, clean_waveform
    

