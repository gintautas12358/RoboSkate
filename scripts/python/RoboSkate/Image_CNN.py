"""
RoboSkate classify images to driving direction

The expert data with the steering angle is used to train a model that extracts a steering angle from the images.
No good results have been achieved yet.
"""

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T


import os
from torchvision.io import read_image
import pickle


# --------------------------------------------------------------------------------
# ------------------ Dataloder -------- ------------------------------------------
# --------------------------------------------------------------------------------

class RoboSkateImageDataset():

    def loadImageList(self):
        with open("./expert_data/Steering_angle/image_data.pkl", "rb") as fp:  # Pickling
            img_labels = pickle.load(fp)
        self.img_labels = img_labels
        print("Bilder: " + str(len(img_labels)))

    def __init__(self, transform=None, target_transform=None):
        self.loadImageList()
        self.img_dir = "./expert_data/Steering_angle/images/"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)-1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = read_image(img_path)
        label = self.img_labels[idx][1]

        autocontraster = T.RandomAutocontrast()
        image = autocontraster(image)
        cropper = T.RandomResizedCrop(size=(80, 80))
        image = cropper(image)

        return image, label






# --------------------------------------------------------------------------------
# ------------------ Model -------------------------------------------------------
# --------------------------------------------------------------------------------


# Define model
class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6*6*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        


    # Defining the forward pass
    def forward(self, x):
        #print(x.size())
        x = self.cnn_layers(x.float())
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.flatten(x)
        #print(x.size())
        logits = self.linear_relu_stack(x)
        #print(logits.size())
        logits = torch.squeeze(logits)
        return logits.double()


# --------------------------------------------------------------------------------
# ------------------ Train loop --------------------------------------------------
# --------------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).double()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Training loss: {loss:>7f} ")




# --------------------------------------------------------------------------------
# ------------------ Test loop --------------------------------------------------
# --------------------------------------------------------------------------------
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Test Avg loss: {test_loss:>8f} \n")

# --------------------------------------------------------------------------------
# ------------------ Script --------------------------------------------------
# --------------------------------------------------------------------------------

# create dataset
print("load dataset")
dataset = RoboSkateImageDataset()

batch_size = 64
test_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and test splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)






# Display one image and label.
'''
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze().permute(1,2,0)
label = train_labels[0]
plt.imshow(img)
plt.title(f"Label: {label}")
plt.show()
print(f"Label: {label}")
'''



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# create Model
model = CNNNetwork().to(device)
#print(model)

# define lossfunktion and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-6)

if os.path.isfile("./scripts/python/RoboSkate/agent_models/CNN/model.pth"):
    # load previously created model trained model
    model.load_state_dict(torch.load("./scripts/python/RoboSkate/agent_models/CNN/model.pth"))
    print("Found previously model.")


# learning
epochs = 5000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")



torch.save(model.state_dict(), "./scripts/python/RoboSkate/agent_models/CNN/model.pth")
print("Saved PyTorch Model State to ./scripts/python/RoboSkate/agent_models/CNN/model.pth")