import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import os
import wandb
from dataset.cifar10 import CIFAR10Dataset
from model.resnet import *

List_num_channels = [
    [64, 128, 256, 512],
    [128, 128, 256, 64],
    [128, 128, 128, 128],
    [64, 128, 128, 128],
    [64, 128, 128, 256],
]
List_num_blocks = [
    [1, 1, 1, 1],
    [2, 3, 3, 2],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [3, 3, 2, 3]
]


batch_size = 64
num_epochs = 70
learning_rate = 0.01

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not torch.cuda.is_available():
        print("Warning: GPU is not available!")

    scores = np.empty((len(List_num_blocks), 5))

    for run, (num_channels, num_blocks) in enumerate(zip(List_num_channels, List_num_blocks)):

        print("run", run, ": channels:", ",".join(str(i) for i in num_channels), "blocks:", ",".join(str(i) for i in num_blocks))

        full_dataset = CIFAR10Dataset("cifar-10-batches-py", train=True, transform=transform_train)
        dataset_size = len(full_dataset)  # Should be 50,000 samples
        fold_size = dataset_size // 5  # 5 folds → 10,000 samples each fold
        indices = np.arange(dataset_size)  # Index array [0, 1, ..., 49999]

        for fold in range(5):
            print(f"Starting fold {fold+1}/5")
            # Define validation indices (current fold)
            val_indices = indices[fold * fold_size: (fold + 1) * fold_size]

            # Define training indices (all except current fold)
            train_indices = np.setdiff1d(indices, val_indices)

            # Create training & validation subsets
            train_subset = Subset(full_dataset, train_indices)
            val_subset = Subset(full_dataset, val_indices)
    
            trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
            valloader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2)

            model = ResNet_Custom(BasicBlock, num_channels=num_channels, num_blocks = num_blocks)
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            print("Model total parameters:", total_params)

            criterion = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_score = -1

            # Train
            save_dir = os.path.join("checkpoint", f"cv1_run_{run}_fold_{fold}")
            os.makedirs(save_dir, exist_ok=True)
            print("Training Run:", f"cv1_run_{run}_fold_{fold}")

            wandb.init(
                # Set the project where this run will be logged
                project="ResNet 5M",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=f"cv1_run_{run}_fold_{fold}",
                # Track hyperparameters and run metadata
                config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "num_channels": ",".join(str(i) for i in num_channels),
                "num_blocks": ",".join(str(i) for i in num_blocks),
                "num_parameters": total_params,
                "fold": fold+1
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
        
                        # Forward pass
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

            scores[run, fold] = best_score
            wandb.finish()
    
    np.savetxt("scores_cv1.csv", scores, delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    train()