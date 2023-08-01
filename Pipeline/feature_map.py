import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the neural network model for graph classification
class GraphClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GraphClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define a custom dataset class for your input graphs
class GraphDataset(Dataset):
    def __init__(self, graph_attributes, labels):
        self.graph_attributes = graph_attributes
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        attributes = self.graph_attributes[index]
        label = self.labels[index]
        return attributes, label

# Set up your input tensor and class labels
graph_attributes = [...]  # List of graph attributes
labels = [...]  # List of corresponding class labels

# Create instances of the dataset and dataloader
dataset = GraphDataset(graph_attributes, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the hyperparameters and model configuration
input_size = len(graph_attributes[0])
hidden_size = 64
num_classes = 4
learning_rate = 0.001
num_epochs = 50

# Create an instance of the graph classifier model
model = GraphClassifier(input_size, hidden_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for attributes, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(attributes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Evaluate the model on the test set (if available)
# test_outputs = model(test_attributes)
# predicted_labels = torch.argmax(test_outputs, dim=1)
# ...

# Save the trained model
torch.save(model.state_dict(), 'graph_classifier_model.pth')
