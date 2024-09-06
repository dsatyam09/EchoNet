import os

# Configuration settings
class Config:
    # General settings
    seed = 42
    log_dir = "logs/simclr"
    graph_dir = "Graphs"
    
    # Training settings
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    temperature = 0.5

    # Model settings
    model_name = 'resnet50'  # Options: 'resnet50', 'vgg16', 'custom'
    projection_dim = 128

    # Data settings
    image_size = 224
    num_classes = 3

# Ensure the graph directory exists
if not os.path.exists(Config.graph_dir):
    os.makedirs(Config.graph_dir)
