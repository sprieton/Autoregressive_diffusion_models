# main.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from model.models import LeNetWithTime, TinyTimeViT
from model.train_OA_ARDMs import Trainer
import utils.config as cfg
from datetime import datetime


# 1. Prepare MNIST dataset
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        return (tensor > self.threshold).float()

# Compose transforms: ToTensor converts PIL image to [0,1] float tensor, Binarize thresholds it
transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize(threshold=0.5)  # You can adjust the threshold if needed
])

# Load MNIST dataset with binarization applied
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
trainset = DataLoader(train_dataset, batch_size=cfg.bach_size, shuffle=True)
testset = DataLoader(test_dataset, batch_size=cfg.bach_size, shuffle=False)


# 2. Configurate the model, optimizer and trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = [TinyTimeViT().to(device), LeNetWithTime().to(device)]
model_names = ['LeNetWithTime', 'TinyTimeViT']

for model, name in zip(models, model_names):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_loader=trainset,
        val_loader=testset,
        test_loader=None,
        num_classes=cfg.num_clases,
        device=device
    )

    # 3. Train
    epochs = cfg.num_epochs
    history = trainer.fit(epochs)

    # 4. Save results on a CSV
    records = []
    for epoch, (train_loss, val_loss_per_digit) in enumerate(zip(history['train_loss'], history['val_loss_per_digit']), 1):
        row = {'epoch': epoch, 'train_loss': train_loss}
        for digit in range(10):
            row[f'val_loss_digit_{digit}'] = val_loss_per_digit.get(digit, None)
        records.append(row)

    df = pd.DataFrame(records)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'{cfg.results_dir}/{name}training_history.csv_{time_stamp}', index=False)
    print("Training history saved to taining_history.csv")