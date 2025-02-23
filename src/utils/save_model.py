import torch
from src import MODEL_PATH

def save_model(state):
    torch.save(state, MODEL_PATH)
    print("Model has been saved successfully")
