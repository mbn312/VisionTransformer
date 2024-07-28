import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, lr_scheduler
from data.data_utils import get_config, get_dataset
from models.model import VisionTransformer

DEVICE = torch.device("cpu")

def train_model(config):

    train_loader = get_dataset(config)

    model = VisionTransformer(
        config.d_model,         
        config.n_classes,               
        config.img_size,          
        config.patch_size,        
        config.n_channels,       
        config.n_heads,         
        config.n_layers,         
        config.learned_pe,  
        config.dropout,      
        config.r_mlp,          
        config.bias         
    ).to(DEVICE)

    if config.weight_decay == 0:
        optimizer = Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.epochs - config.warmup_epochs), eta_min=config.lr_min)

    if config.warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.warmup_epochs), end_factor=1.0, total_iters=(config.warmup_epochs - 1), last_epoch=-1)

    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.epochs):
        training_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        if epoch < config.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        print(f'[Epoch {epoch + 1}/{config.epochs}] Training Loss: {training_loss  / len(train_loader) :.3f}')

    return model


def test(model, config):

    test_loader = get_dataset(config, train=False)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    print(f'-------------------------\n Model Accuracy: {(100 * correct / total):.2f} %\n-------------------------')


if __name__=="__main__":
    print("Implemented Datasets: mnist, fashion_mnist, cifar10")
    dataset_name = input("Enter Dataset: ")
    config = get_config(dataset_name.lower())

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", DEVICE, f"({torch.cuda.get_device_name(DEVICE)})" if torch.cuda.is_available() else "")

    model = train_model(config)

    test(model, config)