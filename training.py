import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, lr_scheduler
from data.data_utils import *
from models.model import VisionTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(config):

    train_loader, val_loader, config.train_mean, config.train_std = get_train_val_split(config)

    model = VisionTransformer(
        config.d_model,         
        config.n_classes,               
        config.img_size,          
        config.patch_size,        
        config.n_channels,
        config.mlp_hidden,       
        config.n_heads,         
        config.n_layers,          
        config.learned_pe,  
        config.dropout,               
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

    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        training_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        training_loss = training_loss / len(train_loader)
        # Update learning rate scheduler
        if epoch < config.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        # Validation
        validation_loss = 0.0
        if len(val_loader) > 0:
            model.eval()
            validation_loss = 0.0
            correct, total = 0, 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                if config.get_val_accuracy:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.shape[0]
                    correct += (predicted == labels).sum().item()

            validation_loss = validation_loss / len(val_loader)

        # Saves model if it performed better than the previous best
        if validation_loss <= best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), config.model_location)

        # Print out metrics
        if len(val_loader) <= 0:
            print(f"[Epoch {epoch + 1}/{config.epochs}] Training Loss: {training_loss:.3f}")
        elif config.get_val_accuracy:
            print(f"[Epoch {epoch + 1}/{config.epochs}] Training Loss: {training_loss:.3f} | Validation Loss: {validation_loss:.3f} | Validation Accuracy: {100 * correct / total:.2f}")
        else:
            print(f"[Epoch {epoch + 1}/{config.epochs}] Training Loss: {training_loss:.3f} | Validation Loss: {validation_loss:.3f}")

    return config

def get_model_accuracy(config):
    # Load trained model
    model = VisionTransformer(
        config.d_model,         
        config.n_classes,               
        config.img_size,          
        config.patch_size,        
        config.n_channels,
        config.mlp_hidden,       
        config.n_heads,         
        config.n_layers,         
        config.learned_pe,  
        config.dropout,                
        config.bias         
    ).to(DEVICE)

    model.load_state_dict(torch.load(config.model_location, map_location=DEVICE))

    # Get test set
    test_loader = get_test_set(config)

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
    parser = argparse.ArgumentParser(
        prog="Vision Transformer",
        description="Image classification using a vision transformer"
    )
    parser.add_argument("-d", "--dataset", type=str.lower, choices=["mnist", "fashion_mnist", "cifar10"], default="mnist", help="Name of dataset to use")
    parser.add_argument("-is", "--img_size", type=int, nargs="*", help="Size of dataset images. Input as: height width")
    parser.add_argument("-ps", "--patch_size", type=int, nargs="*", help="Size of patches. Input as: height width")
    parser.add_argument("-dm", "--d_model", type=int, help="Width of model")
    parser.add_argument("-mh", "--mlp_hidden", type=int, help="Width of hidden MLP")
    parser.add_argument("-nh", "--heads", type=int, help="Number of attention heads")
    parser.add_argument("-l", "--layers", type=int, help="Number of encoder layers")
    parser.add_argument("-lp", "--learned_pe", type=bool, choices=[True, False], help="Whether or not to learn positional encodings")
    parser.add_argument("-do", "--dropout", type=float, help="Dropout rate")
    parser.add_argument("-b", "--bias", type=bool, choices=[True, False], help="Bias of linear layers")
    parser.add_argument("-ph", "--prob_hflip", type=float, help="Probability of horizontal flip")
    parser.add_argument("-cp", "--crop_padding", type=int, help="Random crop padding")
    parser.add_argument("-tv", "--train_val_split", type=int, nargs="*", help="Training and validation split sizes. Input as: train_len val_len")
    parser.add_argument("-va", "--get_val_accuracy", type=bool, choices=[True, False], help="Get validation accuracy")
    parser.add_argument("-bs", "--batch_size", type=int, help="Size of batches")
    parser.add_argument("-w", "--workers", type=int, help="Number of workers to use for each DataLoader")
    parser.add_argument("-lr", "--lr", type=float, help="Starting learning rate")
    parser.add_argument("-lm", "--lr_min", type=float, help="Minimum learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train on")
    parser.add_argument("-we", "--warmup_epochs", type=int, help="Number of epochs to warmup learning rate scheduler")
    parser.add_argument("-ml", "--model_location", type=str, help="Location of model file")
    args = parser.parse_args()
    
    config = get_config(args)

    print("Using device: ", DEVICE, f"({torch.cuda.get_device_name(DEVICE)})" if torch.cuda.is_available() else "")

    config = train_model(config)

    get_model_accuracy(config)