import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import wandb
from dataset.cifar10 import CIFAR10Dataset
from model.resnet import *

model_path = os.path.join("checkpoint","cv2_run_1_fold_1","best_model_epoch.pth")

num_channels = [128, 128, 128, 128]
num_blocks = [4, 4, 4, 4]

preact = True
dropout = 0.5
batch_size = 64
num_epochs = 300
learning_rate = 0.0005

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.1,contrast = 0.1,saturation = 0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor = 2,p = 0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomErasing(p=0.2,scale=(0.02, 0.1),value=1.0, inplace=False)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(3407)
    np.random.seed(3407)
    if not torch.cuda.is_available():
        print("Warning: GPU is not available!")
    else:
        torch.cuda.manual_seed_all(3407) # magic seed

    print("resume", ": channels:", ",".join(str(i) for i in num_channels), "blocks:", ",".join(str(i) for i in num_blocks))

    trainset = CIFAR10Dataset("cifar-10-batches-py", train=True, transform=transform_train)
    valset = CIFAR10Dataset("cifar-10-batches-py", train=False, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=2)

    model = ResNet_Custom(BasicBlock, num_channels=num_channels, num_blocks = num_blocks, num_classes=10, preact=preact, dropout=dropout)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Resumed paremeters from:", model_path)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Model total parameters:", total_params)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_score = -1

    # Train
    save_dir = os.path.join("checkpoint", f"retrained")
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        # Set the project where this run will be logged
        project="ResNet 5M",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"retrain",
        # Track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "num_channels": ",".join(str(i) for i in num_channels),
        "num_blocks": ",".join(str(i) for i in num_blocks),
        "num_parameters": total_params,
        "preact": preact,
        "dropout": dropout
    })

    for epoch in range(num_epochs):
        model.train()
        # pdb.set_trace()
        train_loss = 0.0
        
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(img)
            # pdb.set_trace()
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)  # Accumulate training loss

        # Calculate average training loss for the epoch
        train_loss /= len(trainloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        
        with torch.no_grad():
            sum_correct = 0
            for img, label in valloader:
                img, label = img.to(device), label.to(device)

                output = model(img)
                
                # Count correct
                yhat = output.argmax(dim=1)
                sum_correct += (yhat==label).sum().item()

        # Calculate accuracy
        score = sum_correct/len(valloader.dataset)
        print(f"Validation Accuracy: {score:.4f}")

        wandb.log({"acc": score, "loss": train_loss})

        # Save checkpoint if best acc score
        if score > best_score: 
            best_score = score
            if epoch>=10: # save only when there's enough epochs
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))
            print(f"Best model saved with Acc Score: {best_score:.4f}")
            wandb.run.summary["best score"] = score
            wandb.run.summary["epoch of best"] = epoch+1

        # Save checkpoint for each 20 epochs
        if (epoch % 20) == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
            print("epoch:", epoch+1, 
                "best epoch:", wandb.run.summary.get("epoch of best", "N/A"), 
                "acc:", wandb.run.summary.get("best score", "N/A"))

    wandb.finish()


if __name__ == "__main__":
    train()