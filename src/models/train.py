from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

def train(
        model: Module, 
        epochs: int, 
        data_loader: DataLoader, 
        loss_func: Module, 
        optimiser: Optimizer,
        device: str = "cpu"
        ):

    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        running_loss = 0
        for noisy, clean in data_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            loss = loss_func(output.unsqueeze(1), clean)

            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        print(f"Epoch: {epoch} MeanLoss: {running_loss / len(data_loader)}")

    return model.state_dict()

