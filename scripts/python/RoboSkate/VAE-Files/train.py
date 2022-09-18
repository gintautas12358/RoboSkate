"""
Train a VAE model using saved images in a folder
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds
from tqdm import tqdm
import tensorflow as tf

from controller import VAEController
from data_loader import DataLoader
# from torch.utils.data import DataLoader
from model import ConvVAE


def train_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Path to a folder containing images for training', type=str,
                        default='C:/Users/meric/Desktop/TUM/CBMLR/Repo/G1_RoboSkate/scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/images')
    parser.add_argument('--z-size', help='Latent space', type=int, default=256)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-samples', help='Max number of samples', type=int, default=-1)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-4)
    parser.add_argument('--kl-tolerance', help='KL tolerance (to cap KL loss)', type=float, default=0.5)
    parser.add_argument('--beta', help='Weight for kl loss', type=float, default=1.0)
    parser.add_argument('--n-epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--verbose', help='Verbosity', type=int, default=1)
    args = parser.parse_args()

    set_global_seeds(args.seed)

    if not args.folder.endswith('/'):
        args.folder += '/'

    vae = ConvVAE(z_size=args.z_size,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  kl_tolerance=args.kl_tolerance,
                  beta=args.beta,
                  is_training=True,
                  reuse=False)

#    vae_validation = ConvVAE(z_size=args.z_size,
#                             batch_size=args.batch_size,
#                             learning_rate=args.learning_rate,
#                             kl_tolerance=args.kl_tolerance,
#                             beta=args.beta,
#                             is_training=False,
#                             reuse=False)

    images = [im for im in os.listdir(args.folder) if im.endswith('.png')]  # Changed from .jpg
    images = np.array(images)
    n_samples = len(images)

    if args.n_samples > 0:
        n_samples = min(n_samples, args.n_samples)

    print("{} images".format(n_samples))

    # indices for all time steps where the episode continues
    indices = np.arange(n_samples, dtype='int64')
    np.random.shuffle(indices)

    # split indices into minibatches. minibatchlist is a list of lists; each
    # list is the id of the observation preserved through the training
    minibatchlist = [np.array(sorted(indices[start_idx:start_idx + args.batch_size]))
                     for start_idx in range(0, len(indices) - args.batch_size + 1, args.batch_size)]

    data_loader = DataLoader(minibatchlist, images, n_workers=2, folder=args.folder)

    vae_controller = VAEController(z_size=args.z_size)
    vae_controller.vae = vae
#    vae_controller_validation = VAEController(z_size=args.z_size)
#    vae_controller_validation.vae = vae_validation

    writer = tf.summary.FileWriter(
        'C:/Users/meric/Desktop/TUM/CBMLR/Repo/G1_RoboSkate/scripts/python/RoboSkateIL/VAE/logs/tensorboard',
        vae.sess.graph)

    for epoch in range(args.n_epochs):
        pbar = tqdm(total=len(minibatchlist))
        train_loss_avg = []
        r_loss_avg = []
        kl_loss_avg = []

        for obs in data_loader:
            feed = {vae.input_tensor: obs}
            (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
                vae.loss,
                vae.r_loss,
                vae.kl_loss,
                vae.global_step,
                vae.train_op
            ], feed)
            pbar.update(1)
            train_loss_avg.append(train_loss)
            kl_loss_avg.append(kl_loss)
            r_loss_avg.append(r_loss)

        pbar.close()
        train_loss_avg = np.mean(train_loss_avg)
        kl_loss_avg = np.mean(kl_loss_avg)
        r_loss_avg = np.mean(r_loss_avg)

        print("Epoch {:3}/{}".format(epoch + 1, args.n_epochs))
        print("VAE: optimization step", (train_step + 1), train_loss_avg, r_loss_avg, kl_loss_avg)
        summary_train_loss = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss_avg)])
        summary_r_loss = tf.Summary(value=[tf.Summary.Value(tag="r_loss", simple_value=r_loss_avg)])
        summary_kl_loss = tf.Summary(value=[tf.Summary.Value(tag="kl_loss", simple_value=kl_loss_avg)])
        writer.add_summary(summary_r_loss, epoch)
        writer.add_summary(summary_kl_loss, epoch)
        writer.add_summary(summary_train_loss, epoch)

        # Update params
        vae_controller.set_target_params()
        # Load test image
        img_save_path = 'C:/Users/meric/Desktop/TUM/CBMLR/Repo/G1_RoboSkate/scripts/python/RoboSkateIL/VAE/logs/images'
        if args.verbose >= 1:
            for idx in range(5):
                image_idx = np.random.randint(n_samples)
                image_path = args.folder + images[image_idx]
                image = cv2.imread(image_path)
                image = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
                # ROI will not be used.
                # r = ROI
                # im = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                encoded = vae_controller.encode(image)
                reconstructed_image = vae_controller.decode(encoded)[0]
                # Plot reconstruction
                cv2.imshow("Original", image)
                cv2.imwrite(os.path.join(img_save_path, "Original_Image_Epoch" + str(epoch) + "_" + str(idx) + ".png"),
                            image)
                cv2.imshow("Reconstruct", reconstructed_image)
                cv2.imwrite(
                    os.path.join(img_save_path,
                                 "Reconstructed_Image_Epoch" + str(epoch) + "_" + str(idx) + ".png"),
                    reconstructed_image)
                cv2.waitKey(1)

    save_path = "logs/vae-{}/150Epochs".format(args.z_size)
    os.makedirs(save_path, exist_ok=True)
    print("Saving to {}".format(save_path))
    vae_controller.set_target_params()
    vae_controller.save(save_path)


if __name__ == '__main__':
    train_vae()
