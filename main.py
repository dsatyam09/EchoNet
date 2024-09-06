import argparse
import torch  # Make sure this line is at the top
from train_simclr import train_simclr
from fine_tune import fine_tune

def main():
    parser = argparse.ArgumentParser(description='SimCLR Training and Fine-Tuning')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'fine_tune'], required=True, help='Mode to run: pretrain or fine_tune')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.mode == 'pretrain':
        print("Starting SimCLR pretraining...")
        train_simclr(device=device)
    elif args.mode == 'fine_tune':
        print("Starting fine-tuning...")
        fine_tune()

if __name__ == "__main__":
    main()
