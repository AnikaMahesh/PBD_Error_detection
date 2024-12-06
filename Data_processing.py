import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# Update the dataset path to the correct directory (with raw string format)
DATASET_DIR = r"C:\Users\ntiwari\OneDrive - Olin College of Engineering\Desktop\PCB_DATASET"

subfolders = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
labels = []

# Define the images directory
images_dir = os.path.join(DATASET_DIR, 'images')

# Preprocess images
IMAGE_SIZE = (224, 224)
images = []

# Iterate through each subfolder to load and preprocess images
for subfold, subfolder in enumerate(subfolders):
    folder_path = os.path.join(images_dir, subfolder)
    print(f"Trying to access: {folder_path}")
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"Directory found: {folder_path}")  # Directory exists, proceed with loading images
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Loading file: {file}")  # Debug line to check image loading
                image = cv2.imread(file_path)
                if image is not None:
                    # Resize and normalize the image
                    image = cv2.resize(image, IMAGE_SIZE)
                    image = image.astype('float32') / 255.0  # Normalize image pixels
                    images.append(image)
                    labels.append(subfold)
                else:
                    print(f"Failed to read image: {file_path}")
    else:
        print(f"Directory not found: {folder_path}")

# Check if images and labels were loaded
print(f"Total images loaded: {len(images)}")
print(f"Total labels: {len(labels)}")

# If no images are loaded, print an error
if len(images) == 0 or len(labels) == 0:
    print("No images or labels loaded. Check dataset path or image format.")
else:
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Perform one-hot encoding on the labels
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(subfolders))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Optionally: Save the preprocessed data into numpy files for later use
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    print("Data preprocessing completed successfully.")

# Example: Loading and verifying the saved data for both training and testing
X_train_loaded = np.load('X_train.npy')
y_train_loaded = np.load('y_train.npy')
X_test_loaded = np.load('X_test.npy')
y_test_loaded = np.load('y_test.npy')

# Print the shapes of the loaded data
print(f"Loaded training data shape: {X_train_loaded.shape}")
print(f"Loaded training labels shape: {y_train_loaded.shape}")
print(f"Loaded test data shape: {X_test_loaded.shape}")
print(f"Loaded test labels shape: {y_test_loaded.shape}")

# Exploratory Data Analysis
print(f"Total number of images: {images.shape[0]}")
print(f"Image dimensions: {images.shape[1:]}")
print(f"Number of classes: {len(subfolders)}")

# Visualization of random images
def display_sample(images, labels, subfolders, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    random_indices = random.sample(range(len(images)), num_samples)

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(images[idx])  # Display the image
        ax.set_title(subfolders[np.argmax(labels[idx])])  # Show the class name based on the label
        ax.axis('off')

    plt.show()

# Display random images from the dataset
display_sample(X_train, y_train, subfolders)

# Number of images in each class
import matplotlib.pyplot as plt

# Count the occurrences of each label in the dataset
class_counts = np.sum(y_train, axis=0)  # Sum each class label across the training set
plt.bar(subfolders, class_counts)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Training Data')
plt.xticks(rotation=45)
plt.show()

# Evaluating brightness and intensity difference across R-G-B
# Flatten all images to 1D array to calculate pixel intensity distribution
all_images = X_train.reshape(-1, 3)  # Flatten into a 2D array (total_pixels x 3)
plt.figure(figsize=(10, 6))
plt.hist(all_images, bins=50, range=(0, 1), density=True, alpha=0.7, label=['Red', 'Green', 'Blue'])
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.title('Pixel Intensity Distribution')
plt.legend()
plt.show()

# Variability computation pre-augmentation
# Compute the mean pixel value for each image
image_means = np.mean(X_train, axis=(1, 2, 3))  # Average across the 224x224 pixels and the 3 channels
plt.hist(image_means, bins=50, range=(0, 1))
plt.xlabel('Mean pixel value')
plt.ylabel('Frequency')
plt.title('Histogram of Image Mean Pixel Values before Image Augmentation')
plt.show()

def augment_image(image):
    # Random flip (horizontal or vertical)
    flip_type = random.choice([0, 1])  # 0: vertical, 1: horizontal
    flipped_image = cv2.flip(image, flip_type)

    # Adjust brightness using a random factor (gamma correction)
    gamma = random.uniform(0.5, 1.5)  # Random brightness factor
    bright_image = np.power(flipped_image, gamma)  # Apply gamma correction directly

    # Ensure values are in the range [0, 1]
    bright_image = np.clip(bright_image, 0, 1)

    # Add Gaussian noise
    row, col, ch = bright_image.shape
    gauss = np.random.normal(0, 0.1, (row, col, ch))  # Mean = 0, Standard deviation = 0.1
    noisy_image = bright_image + gauss
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure values are in the range [0, 1]

    return noisy_image



# Apply augmentation to a subset of images (200-300 images)
augmented_images = []
augmented_labels = []

num_samples = random.randint(200, 300)  # Randomly pick between 200-300 images to augment
sample_indices = random.sample(range(len(images)), num_samples)

for idx in sample_indices:
    image = images[idx]
    label = labels[idx]
    augmented_image = augment_image(image)
    augmented_images.append(augmented_image)
    augmented_labels.append(label)

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Combine original images and augmented images
X_combined = np.concatenate([images, augmented_images], axis=0)
y_combined = np.concatenate([labels, augmented_labels], axis=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Save the augmented data into numpy files
np.save('X_train_augmented.npy', X_train)
np.save('y_train_augmented.npy', y_train)
np.save('X_test_augmented.npy', X_test)
np.save('y_test_augmented.npy', y_test)

print("Data augmentation completed successfully.")

# Example: Visualizing some augmented images
def display_sample(images, labels, subfolders, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    random_indices = random.sample(range(len(images)), num_samples)
    
    for i, idx in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(images[idx])  # Display the image
        ax.set_title(subfolders[np.argmax(labels[idx])])  # Show the class name based on the label
        ax.axis('off')
    
    plt.show()

# Display random images from the augmented dataset
display_sample(X_train, y_train, subfolders)

# Variability computation post-augmentation
image_means = np.mean(X_train, axis=(1, 2, 3))  # Average across the 224x224 pixels and the 3 channels
plt.hist(image_means, bins=50, range=(0, 1))
plt.xlabel('Mean pixel value')
plt.ylabel('Frequency')
plt.title('Histogram of Image Mean Pixel Values after Augmentation')
plt.show()


#Model Deployment
import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
from typing import List

class PCBDefectCNN(nn.Module):
    def __init__(self, num_classes):
        super(PCBDefectCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flattening the tensor before feeding into fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Model Architecture (CNN)
class PCBDefectCNN(nn.Module):
    def __init__(self, num_classes):
        super(PCBDefectCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flattening the tensor before feeding into fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Preparing the data for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
num_classes = len(subfolders)
model = PCBDefectCNN(num_classes=num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects class indices (not one-hot encoded)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
train_acc = []
test_acc = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)  # model outputs logits
        loss = criterion(outputs, labels)  # Using CrossEntropyLoss
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds * 100
    train_acc.append(epoch_acc)

    # Evaluate on test set
    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    
    test_acc_epoch = correct_preds / total_preds * 100
    test_acc.append(test_acc_epoch)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc_epoch:.2f}%")

# Plot training and test accuracy
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), "pcb_defect_model.pth")

# Error Analysis using CrossEntropyLoss
model.eval()
correct_preds = 0
total_preds = 0
class_wise_errors = np.zeros(num_classes)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        for idx, label in enumerate(labels):
            if predicted[idx] != label:
                class_wise_errors[label] += 1
        
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

test_acc = correct_preds / total_preds * 100
print(f"Test Accuracy: {test_acc:.2f}%")
