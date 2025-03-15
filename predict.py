import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from dataset.cifar10 import CIFAR10Dataset
from model.resnet import *
import os

model_path = os.path.join("checkpoint","retrained","best_model.pth")
output_csv = "prediction.csv"

num_channels = [128, 128, 128, 128]
num_blocks = [4, 4, 4, 4]

preact = True # Pre-activation blocks 
dropout = 0.5 # Dropout is not conducted in prediction
batch_size = 64

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def predict():
    if not torch.cuda.is_available():
        print("Warning: GPU is not available!")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    testset = CIFAR10Dataset("cifar-10-batches-py", train=False, transform=transform_test, predict=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load model
    model = ResNet_Custom(BasicBlock, num_channels=num_channels, num_blocks=num_blocks, preact=preact, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for img in testloader:  # No labels in prediction mode
            img = img.to(device)

            # Forward pass
            output = model(img)
            yhat = output.argmax(dim=1)  # Get class predictions

            # Store predictions
            predictions.extend(yhat.cpu().numpy())

    # Save predictions to CSV
    df = pd.DataFrame({"ID": range(len(predictions)), "Labels": predictions})
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    predict()
