import torch
import torch.nn as nn
from torch.optim import Adam
from data.data_utils import get_dataset
from models.model import VisionTransformer

DEVICE = torch.device("cpu")

def train(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, learned_pe, dropout, r_mlp, bias, batch_size, epochs, lr):

    train_loader = get_dataset(img_size, batch_size)

    vit = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, learned_pe, dropout, r_mlp, bias).to(DEVICE)

    optimizer = Adam(vit.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        training_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = vit(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')

    return vit


def test(vit, img_size, batch_size):

    correct = 0
    total = 0

    test_loader = get_dataset(img_size, batch_size, DEVICE)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = vit(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Model Accuracy: {100 * correct // total} %') 


if __name__=="__main__":
    d_model = 9
    n_classes = 10
    img_size = (32,32)
    patch_size = (16,16)
    n_channels = 1
    n_heads = 3
    n_layers = 3
    learned_pe = True
    dropout = 0.2
    r_mlp = 4
    bias = False
    batch_size = 128
    epochs = 5
    lr = 0.005

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", DEVICE, f"({torch.cuda.get_device_name(DEVICE)})" if torch.cuda.is_available() else "")

    vit = train(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, learned_pe, dropout, r_mlp, bias, batch_size, epochs, lr)

    test(vit, img_size, batch_size)