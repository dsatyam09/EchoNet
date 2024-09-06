import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms  # Add this import
from simclr_model import SimCLR
from dataset_loader import UnlabeledDataset
from loss import NT_Xent_Loss
from config import Config
import visualization
import utils

def train_simclr(device='cpu'):
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust the size if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = UnlabeledDataset(root_dir='dataset/unlabeled', transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    # Model, loss, and optimizer
    model = SimCLR(model_name=Config.model_name, projection_dim=Config.projection_dim).to(device)
    criterion = NT_Xent_Loss(temperature=Config.temperature)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    # Gradient accumulation parameters
    accumulation_steps = 4  # Number of steps to accumulate gradients

    # Training loop
    for epoch in range(1, Config.num_epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            images, _ = batch
            images = images.to(device)
            # Forward pass
            projections = model(images)
            loss = criterion(projections, projections)  # Note: Adjust this line if needed
            loss.backward()

            # Update weights after accumulating gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        # Average loss for this epoch
        epoch_loss /= len(dataloader)
        
        # Save checkpoints
        if epoch % 10 == 0:
            utils.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            })
        
        if epoch % 100 == 0:
            utils.save_best_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            })
        
        # Log training loss
        print(f'Epoch [{epoch}/{Config.num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Save visualizations
        if epoch % 10 == 0:
            # Collect embeddings for visualization
            embeddings, labels = extract_embeddings_and_labels(model, dataloader, device)
            visualization.save_tsne_plot(embeddings, labels, epoch)
            # Example values, replace with actual
            true_labels = []  
            predicted_labels = []  
            y_true = []
            y_pred_prob = []
            visualization.save_confusion_matrix(true_labels, predicted_labels, epoch)
            visualization.save_roc_curve(y_true, y_pred_prob, epoch)
            sample_images = []  # Example, replace with actual images
            augmentations = None  # Replace with actual augmentation pipeline
            visualization.save_augmented_samples(sample_images, augmentations, epoch)
    
    # Save final checkpoint
    utils.save_checkpoint({
        'epoch': Config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    })


def extract_embeddings_and_labels(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            # Forward pass to get embeddings
            features = model.encode(images)  # Assuming 'encode' returns the feature embeddings
            embeddings.append(features.cpu().numpy())  # Convert to numpy array and append
            labels.append(_ if _ is not None else np.array([]))  # Append labels if available

    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0) if len(labels) > 0 else np.array([])

    return embeddings, labels


if __name__ == "__main__":
    train_simclr(device='cuda' if torch.cuda.is_available() else 'cpu')
