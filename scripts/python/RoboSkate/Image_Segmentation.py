"""
RoboSkate classify images to driving direction

https://github.com/sksq96/pytorch-vae

The images with the labels on which the path is marked were used to train a variational autoencoder
whose encoder then serves as image preprocessing for the RL agents.
"""

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as Functional
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter
from random import *
import cv2
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import imageio
from pathlib import Path



z_dim = 2
load_privious_model = True
enjoy_latent = False
create_latent = True
epochs = 100


# --------------------------------------------------------------------------------
# ------------------ enjoy_latent ------------------------------------------------
# https://github.com/araffin/learning-to-drive-in-5-minutes/blob/master/vae/enjoy_latent.py
# --------------------------------------------------------------------------------


def create_figure_and_sliders(name, state_dim):
    """
    Creating a window for the latent space visualization,
    and another one for the sliders to control it.
    :param name: name of model (str)
    :param state_dim: (int)
    :return:
    """
    # opencv gui setup
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(name, 500, 500)
    cv2.namedWindow('slider for ' + name)
    print("show window")
    # add a slider for each component of the latent space
    for i in range(state_dim):
        # the sliders MUST be between 0 and max, so we placed max at 100, and start at 50
        # So that when we substract 50 and divide 10 we get [-5,5] for each component
        cv2.createTrackbar(str(i), 'slider for ' + name, 50, 100, (lambda a: None))


# --------------------------------------------------------------------------------
# ------------------ Dataloder -------- ------------------------------------------
# --------------------------------------------------------------------------------

def my_segmentation_transform(input, target):
    i, j, h, w = transforms.RandomResizedCrop.get_params(input, scale=(0.5, 1.0), ratio=(3, 3.5))
    input = F.crop(input, i, j, h, w)
    target = F.crop(target, i, j, h, w)
    if random.random() > 0.5:
        input = F.hflip(input)
        target = F.hflip(target)
    return input, target


class RoboSkateImageDataset():

    def __init__(self):
        self.img_dir = "./expert_data/Segmentation/"
        self.images = [im for im in os.listdir(str(self.img_dir) + "/images/") if im.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join((str(self.img_dir) + "images"), self.images[idx])
        image = read_image(img_path)
        lable_path = os.path.join((str(self.img_dir) + "images_ground_truth"), self.images[idx])
        label = read_image(lable_path)

        image, label = my_segmentation_transform(image, label)

        scaledown = T.Resize((60, 200))
        image = scaledown(image)
        grayskale = T.Grayscale()
        label = grayskale(scaledown(label))


        # Normalization
        image = image.to(torch.float) / 255
        label = label.to(torch.float) / 255

        return image, label


# --------------------------------------------------------------------------------
# ------------------ Model -------------------------------------------------------
# --------------------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 10)


# Define model
class SegmentationNetwork(nn.Module):
    def __init__(self, image_channels=3, h_dim=2560, z_dim=z_dim):
        super(SegmentationNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(8, 6), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        '''
        cv2.imshow("original imagebig", x.cpu().detach().numpy()[0][0])
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            pass
        '''
        # print(x.shape) #(14x 3x 200x60)
        h = self.encoder(x)
        # print(h.shape) (14x 2560)
        z, mu, logvar = self.bottleneck(h)
        # print(z.shape) # (14x 32)
        # print(mu.shape) # (14x 32)
        z = self.fc3(z)
        # print(z.shape) #(14x 2560)
        output = self.decoder(z)
        # print(output.shape)  # (14x 3x 200x 60)
        return output, mu, logvar

    def decode_from_z(self, z):
        # print(z)
        z = self.fc3(z)
        # print(z.shape) #(14x 2560)
        output = self.decoder(z)
        # print(output.shape)  # (14x 3x 200x 60)
        return output


# --------------------------------------------------------------------------------
# ------------------ Train loop --------------------------------------------------
# --------------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, TensorBoard, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, mu, logvar = model(X)
        loss = loss_fn(pred, y, mu, logvar)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            TensorBoard.add_scalar("tain/loss", loss / size, epoch)


# --------------------------------------------------------------------------------
# ------------------ Test loop --------------------------------------------------
# --------------------------------------------------------------------------------
def test(dataloader, model, TensorBoard, epoch):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred, mu, logvar = model(X)
            test_loss += loss_fn(pred, y, mu, logvar)

    y_rgb = torch.cat((y, y, y), 1)
    pred_rgb = torch.cat((pred, pred, pred), 1)
    all_images = torch.cat((X[0].unsqueeze(0), y_rgb[0].unsqueeze(0), pred_rgb[0].unsqueeze(0)), 0)
    print(X.shape[0])
    for i in range(1, X.shape[0]):
        all_images = torch.cat((all_images, X[i].unsqueeze(0), y_rgb[i].unsqueeze(0), pred_rgb[i].unsqueeze(0)), 0)
    grid_all = torchvision.utils.make_grid(all_images, 3)
    TensorBoard.add_image('test/input+ground_truth+prediction', grid_all, epoch)

    TensorBoard.add_scalar("test/loss", test_loss / size, epoch)


# --------------------------------------------------------------------------------
# ------------------ Loss --------------------------------------------------
# --------------------------------------------------------------------------------

def loss_fn(pred, x, mu, logvar):
    BCE = Functional.binary_cross_entropy(pred, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE + KLD


# --------------------------------------------------------------------------------
# ------------------ Script --------------------------------------------------
# --------------------------------------------------------------------------------

# Setup Tensorbord
number_of_run = 1
while os.path.exists("./scripts/python/RoboSkate/logs/Segmentation_" + str(number_of_run) + "/"):
    number_of_run += 1
TensorBoard = SummaryWriter('./scripts/python/RoboSkate/logs/Segmentation_' + str(number_of_run) + '/')
print("TensorBoard log under: ./scripts/python/RoboSkate/logs/Segmentation_" + str(number_of_run) + "/")

# create dataset
print("load dataset")
dataset = RoboSkateImageDataset()

batch_size = 64
# test_split = .001
test_split = .01
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and test splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset:
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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# create Model
model = SegmentationNetwork().to(device)
# print(model)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

if os.path.isfile("./trained_models/Segmentation/model/model_z_dim_" + str(z_dim) + "_old_new.pth") and load_privious_model:
    # load previously created model trained model
    model.load_state_dict(torch.load("./trained_models/Segmentation/model/model_z_dim_" + str(z_dim) + "_old_new.pth"))
    print("Found previously model.")

images, labels = next(iter(test_dataloader))
TensorBoard.add_graph(model, images)

# ------------------------------------------------------------------------------------
# enjoy latent space quick implementation
# ------------------------------------------------------------------------------------


if enjoy_latent:
    print("enjoy latent is active. no training!")
    z_size = z_dim
    fig_name = "segmentation_decoder"
    create_figure_and_sliders(fig_name, z_size)

    should_exit = False
    while not should_exit:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        state = []
        for i in range(z_size):
            state.append(cv2.getTrackbarPos(str(i), 'slider for ' + fig_name))
        # Rescale the values to fit the bounds of the representation
        state = (np.array(state) / 100 * 10 - 5).astype(np.float32)
        # print("z-space: " + str(state))

        reconstructed_image = model.decode_from_z(torch.tensor([state])).detach().numpy()[0].transpose([1, 2, 0])

        # stop if user closed a window
        if (cv2.getWindowProperty(fig_name, 0) < 0) or (cv2.getWindowProperty('slider for ' + fig_name, 0) < 0):
            should_exit = True
            break
        cv2.imshow(fig_name, reconstructed_image)

    # gracefully close
    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------
# create a grid with two axis of the latent space
# ------------------------------------------------------------------------------------

if create_latent:
    if not (z_dim == 2):
        print("Create_latent is only implemented for z_dim == 2")

    print("Create_latent is active. no training!")

    frature_1_range = np.arange(-1.6, 1.6, 0.2) # Vertical
    step_size = 0.6
    frature_2_range = np.arange(-3*step_size, 4*step_size, step_size) # Horizontal

    state = (np.array([0, 0])).astype(np.float32)
    reconstructed_image = model.decode_from_z(torch.tensor([state]))
    all_images = reconstructed_image

    for f1 in frature_1_range:
        for f2 in frature_2_range:
            state = (np.array([f1,f2])).astype(np.float32)
            reconstructed_image = model.decode_from_z(torch.tensor([state]))
            all_images = torch.cat((all_images, reconstructed_image), 0)

    grid_all = torchvision.utils.make_grid(all_images[1:], frature_2_range.size)
    imageio.imwrite("./scripts/python/RoboSkate/logs/Segmentation_latent/featurespace.jpg", grid_all.detach().numpy().transpose([1, 2, 0]))



# ------------------------------------------------------------------------------------


# learning
if (not create_latent) and (not enjoy_latent):
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        train(train_dataloader, model, loss_fn, optimizer, TensorBoard, t)
        test(test_dataloader, model, TensorBoard, t)

    torch.save(model.state_dict(), "./scripts/python/RoboSkate/agent_models/Segmentation/new_model.pth")
    print("Saved PyTorch Model State to ./scripts/python/RoboSkate/agent_models/Segmentation/new_model.pth")

plt.close()
TensorBoard.flush()
TensorBoard.close()
