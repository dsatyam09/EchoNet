import torchvision.transforms as transforms

def get_simclr_augmentations():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])
