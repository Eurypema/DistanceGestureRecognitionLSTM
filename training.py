# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Step 1: Read the Logged Data
# Define the directory path containing the log files
log_dir = "C:\\Users\\gordo\\AppData\\Local\\teraterm5"

# Initialize an empty list to hold dataframes
all_data = []

# Function to map file names to labels
def label_from_filename(filename):
    if 'wave' in filename:
        return 0
    elif 'push' in filename:
        return 1
    elif 'pull' in filename:
        return 2
    else:
        return -1

# Iterate through each file in the directory
for file_name in os.listdir(log_dir):
    if file_name.endswith('.log'):  # Check if the file is a log file
        file_path = os.path.join(log_dir, file_name)
        # Read the data into a pandas DataFrame, specifying the column names
        data = pd.read_csv(file_path, names=['timestamp', 'distance', 'infrared_value'])
        # Apply the label to each row based on the filename
        data['target'] = label_from_filename(file_name)
        all_data.append(data)

# Combine all dataframes into one
data = pd.concat(all_data, ignore_index=True)

# Step 2: Preprocess the Data
# Convert distance and infrared_value to numeric, replacing errors with NaN
data['distance'] = pd.to_numeric(data['distance'], errors='coerce')
data['infrared_value'] = pd.to_numeric(data['infrared_value'], errors='coerce')
# Drop rows with any NaN values
data.dropna(inplace=True)

# Calculate the time difference
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data['time_diff'] = data['timestamp'].diff().dt.total_seconds().fillna(0)

# Calculate the rate of change of distance (delta_distance)
data['delta_distance'] = data['distance'].diff().fillna(0)

# Calculate the rate of change of infrared_value
data['delta_infrared'] = data['infrared_value'].diff().fillna(0)

# Drop rows with any NaN values again after new features
data.dropna(inplace=True)

# Ensure all columns are numeric
data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']] = data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']].apply(pd.to_numeric, errors='coerce')

# Normalize the data (including the new features)
data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']] = (data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']] - data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']].mean()) / data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']].std()

# Filter out any rows with undefined labels
data = data[data['target'] != -1]

# Extract features (distance, infrared_value, delta_distance, delta_infrared, time_diff) and labels (target)
X = data[['distance', 'infrared_value', 'delta_distance', 'delta_infrared', 'time_diff']].values.astype(np.float32)
y = data['target'].values

# Data augmentation
def augment_data(X, y):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        original = X[i]
        label = y[i]
        augmented_X.append(original)
        augmented_y.append(label)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, original.shape)
        augmented_X.append(original + noise)
        augmented_y.append(label)
        
        # Scale data
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented_X.append(original * scale_factor)
        augmented_y.append(label)
        
        # Time shift
        shift = np.random.randint(1, 5)
        shifted = np.roll(original, shift)
        augmented_X.append(shifted)
        augmented_y.append(label)
        
        # Flip
        flipped = np.flip(original)
        augmented_X.append(flipped)
        augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)

X_augmented, y_augmented = augment_data(X, y)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Convert features and labels to PyTorch tensors
X_tensor = torch.tensor(X_augmented, dtype=torch.float32)
y_tensor = torch.tensor(y_augmented, dtype=torch.long)

# Step 4: Define a Dataset and DataLoader
# Custom Dataset class for sensor data
class SensorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return a tuple of features and corresponding label
        return self.features[idx], self.labels[idx]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation
train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=32, shuffle=False)

# Step 5: Build and Train a PyTorch Model
# Define an LSTM-based model for gesture recognition
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.bn(lstm_out)
        fc1_out = self.fc1(lstm_out)
        fc1_out = nn.ReLU()(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = nn.ReLU()(fc2_out)
        output = self.fc3(fc2_out)
        return output

# Initialize the model, loss function, and optimizer
input_size = 5  # Five input features: distance, infrared_value, delta_distance, delta_infrared, time_diff
hidden_size = 100  # Number of LSTM hidden units
output_size = 3  # Number of output classes: 'wave', 'push', 'pull'

model = GestureRecognitionModel(input_size, hidden_size, output_size)

model.load_state_dict(torch.load('gesture_recognition_model.pth'))


# Use weighted loss function to handle class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)  # Weight decay for regularization

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Initialize lists to store metrics for each epoch
precision_list = []
recall_list = []
f1_list = []

for epoch in range(30):  # Number of epochs
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        features = features.unsqueeze(1)
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    
    print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping")
        break

# Save the model
torch.save(model.state_dict(), 'gesture_recognition_model.pth')
print("Model saved to gesture_recognition_model.pth")