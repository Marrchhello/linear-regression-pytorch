import torch
from torch import nn
import matplotlib.pyplot as plt

train_losses = []
test_losses = []

# Data generation
start = 0
end = 10
step = 0.2
x = torch.arange(start, end, step)

weight = 9
bias = 5
y = x * weight + bias

# Split data into training and test sets
train = int(0.8 * len(x))
Xtrain, Ytrain = x[:train], y[:train]
Xtest, Ytest = x[train:], y[train:]

# Function to plot predictions
def plotpred(trainData=Xtrain, trainLabels=Ytrain, testData=Xtest, testLabels=Ytest, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(trainData, trainLabels, c="g", s=4, label="Training data")
    plt.scatter(testData, testLabels, c="b", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(testData, predictions, c="r", s=4, label="Predictions")
    plt.legend()
    plt.show()

# Linear regression model
class linRegMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# Initialize model
model0 = linRegMod()

# Print initial model parameters
for param in model0.parameters():
    print(param)

# Loss function and optimizer
lossFunc = nn.L1Loss()
optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model0.train()
    yPred = model0(Xtrain)
    loss = lossFunc(yPred, Ytrain)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())


    # Testing phase after each epoch
    model0.eval()
    with torch.no_grad():
        testPred = model0(Xtest)
        testLoss = lossFunc(testPred, Ytest)

        test_losses.append(testLoss.item())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Training Loss: {loss:.4f} | Test Loss: {testLoss:.4f}")

# Model's learned parameters
print("Model parameters after training:")
print(model0.state_dict())

# Plot final predictions
with torch.no_grad():
    y_preds = model0(Xtest)
plotpred(predictions=y_preds)


def plot_loss(train_losses, test_losses):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Training Loss", c="blue")
    plt.plot(test_losses, label="Test Loss", c="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.show()

plot_loss(train_losses, test_losses)