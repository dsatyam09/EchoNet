import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_loader import LabeledDataset
from simclr_model import SimCLR
from config import Config
from utils import save_checkpoint, load_checkpoint

def fine_tune():
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load labeled dataset
    train_dataset = LabeledDataset(root_dir='dataset/labeled/train', transform=transform)
    test_dataset = LabeledDataset(root_dir='dataset/labeled/test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    # Initialize the SimCLR model with fine-tuning layers
    model = SimCLR(model_name=Config.model_name, projection_dim=Config.projection_dim)
    
    # Define the loss function and optimizer for fine-tuning
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    # Load the pretrained model
    checkpoint = load_checkpoint('checkpoints/best_checkpoint.pth')  # Adjust path as necessary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Fine-tuning loop
    for epoch in range(1, Config.num_epochs + 1):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model.encode(images)  # Use encoder for feature extraction
            logits = model.projection_head(outputs)  # Project features
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Average loss for this epoch
        epoch_loss /= len(train_loader)
        
        print(f'Epoch [{epoch}/{Config.num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Save checkpoints
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            })
        
        # Save final checkpoint
        if epoch == Config.num_epochs:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            })
        
        # Validation phase
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = model.encode(images)
                    logits = model.projection_head(outputs)
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    fine_tune()
