from dataset_loader.dataset import Data
from torch.utils.data import DataLoader

def load_data(clean_dir: str, noisy_dir: str, transform = None):

    dataset = Data(noisy_dir, clean_dir, transform)
    data_loader = DataLoader(dataset, batch_size=16)

    print("Loaded data successfully")

    return data_loader

