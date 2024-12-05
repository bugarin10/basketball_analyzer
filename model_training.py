import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from sklearn.model_selection import train_test_split


class HyperParameters:
    def __init__(self):
        self.train_batch_size = 4
        self.valid_batch_size = 4
        self.lstm_layers = 1
        self.hidden_layer = 200
        self.learning_rate = 0.001
        self.n_epochs = 30
        self.bidirectional = False
        self.dropout = 0.1


def create_dataset():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "02_keypoiunts"
    )
    files = os.listdir(path)
    X = [np.load(file) for file in files if file.endswith(".npy")]
    y = [1 if file.find("make") >= 0 else 0 for file in files if file.endswith(".npy")]

    X_temp, X_valid, y_temp, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42
    )

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_valid),
        torch.tensor(y_valid),
        torch.tensor(X_test),
        torch.tensor(y_test),
    )


# Data Loading
x_train, y_train, x_valid, y_valid, x_test, y_test = create_dataset()

# Hyperparmeters
params = HyperParameters()

# Model Training Tools
optimizer = optim.Adam()  # ADD MODEL PARAMETERS HERE
criterion = nn.CrossEntropyLoss()
train_loader = data.DataLoader(
    data.TensorDataset(x_train, y_train),
    shuffle=True,
    batch_size=params.train_batch_size,
)
validation_loader = data.DataLoader(
    data.TensorDataset(x_valid, y_valid),
    shuffle=False,
    batch_size=params.valid_batch_size,
)
