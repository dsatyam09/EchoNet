import torch

def save_checkpoint(state, filename='checkpoints/last_checkpoint.pth'):
    torch.save(state, filename)

def save_best_checkpoint(state, filename='checkpoints/best_checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(filename='checkpoints/best_checkpoint.pth'):
    return torch.load(filename)
