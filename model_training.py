import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from sklearn.model_selection import train_test_split
from form_analyzer import BallAnalyzer


class HyperParameters:
    def __init__(self):
        self.train_batch_size = 4
        self.valid_batch_size = 4
        self.lstm_layers = 1
        self.input_size = 34 * 3  # Keypoints = 34 and Features per = 3
        self.hidden_layer = 200
        self.learning_rate = 0.001
        self.n_epochs = 5
        self.bidirectional = False
        self.dropout = 0.1


def create_dataset():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "02_keypoiunts"
    )
    files = os.listdir(path)
    X = [np.load(file) for file in files if file.endswith(".npy")]
    y = [1 if file.find("make") >= 0 else 0 for file in files if file.endswith(".npy")]

    # If causes errors, make dtype-object
    X = np.array(X)

    X_temp, X_valid, y_temp, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42
    )

    return (
        torch.tensor(np.array(X_train)),
        torch.tensor(np.array(y_train)),
        torch.tensor(np.array(X_valid)),
        torch.tensor(np.array(y_valid)),
        torch.tensor(np.array(X_test)),
        torch.tensor(np.array(y_test)),
    )


def train_model(
    model, train_loader, valid_loader, criterion, optimizer, n_epochs, device
):
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    epochs = []

    for epoch in range(n_epochs):
        epochs.append(epoch)
        model.train()
        train_loss = 0
        valid_loss = 0
        t_correct = 0
        t_total = 0
        v_correct = 0
        v_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device)

            y_pred = model(inputs)
            loss = criterion(y_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            t_total += targets.size(0)
            t_correct += (predicted == targets).sum().item()

        training_loss.append(train_loss / len(train_loader))
        training_accuracy.append(t_correct / t_total)

        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                y_pred = model(inputs)
                valid_loss += criterion(y_pred, targets).item()
                _, predicted = torch.max(y_pred, 1)
                v_total += targets.size(0)
                v_correct += (predicted == targets).sum().item()

        validation_accuracy(v_correct / v_total)
        validation_loss.append(valid_loss / len(valid_loader))

        print(f"========== Epoch {epoch} ==========")
        print(
            f"Training Loss: {train_loss/len(train_loader)} Training Accuracy: {t_correct/t_total}"
        )
        print(
            f"Validation Loss: {valid_loss/len(valid_loader)} Validation Accuracy: {v_correct/v_total}"
        )

    return (
        training_loss,
        training_accuracy,
        validation_accuracy,
        validation_loss,
        epochs,
    )


def test_model(model, test_loader, criterion, n_epochs, device):
    test_loss = 0
    test_accuracy = 0

    for epoch in range(n_epochs):
        model.eval()
        test_loss = 0
        t_correct = 0
        t_total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device)

                y_pred = model(inputs)
                test_loss += criterion(y_pred, targets).item()

                _, predicted = torch.max(y_pred, 1)
                t_total += targets.size(0)
                t_correct += (predicted == targets).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = t_correct / t_total
        return test_loss, test_accuracy


# Set Random Parameters
torch.manual_seed(42)
np.random.seed(42)

# Data Loading
x_train, y_train, x_valid, y_valid, x_test, y_test = create_dataset()

# Hyperparmeters
params = HyperParameters()

# The Model
model = BallAnalyzer()

# Model Training Tools
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
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
test_loader = data.DataLoader(
    data.TensorDataset(x_test, y_test), shuffle=True, batch_size=params.valid_batch_size
)


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Run on GPU...")
else:
    print("Run on CPU...")

model.to(device)
