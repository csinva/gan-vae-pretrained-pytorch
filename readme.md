<h1 align="center"> Pre-trained GANs, VAEs + classifiers for MNIST / CIFAR10 </h1>
<p align="center"> A simple starting point for modeling with GANs/VAEs in pytorch.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.9-blue">
  <img src="https://img.shields.io/github/checks-status/csinva/gan-vae-pretrained-pytorch/master">
</p>  

- includes model class definitions + training scripts
- includes notebooks showing how to load pretrained nets / use them
- tested with pytorch 1.0+
- generates images the same size as the dataset images

### mnist

Generates images the size of the MNIST dataset (28x28), using an architecture based on the [DCGAN paper](http://arxiv.org/abs/1511.06434). Trained for 100 epochs. Weights [here](https://github.com/csinva/pytorch_gan_pretrained/tree/master/mnist_dcgan/weights).

| data samples                                                |  dcgan samples                                       |    vae samples                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| ![real_images](mnist_dcgan/samples/real_samples.png)  | ![fake_images-300](mnist_dcgan/samples/fake_samples_epoch_099.png) | ![fake_images-300](mnist_vae/samples/sample_25.png) | 

For comparison with a less complicated architecture, I've also included a pre-trained non-convolutional GAN  in the [mnist_gan_mlp](mnist_mlp) folder, based on code from [this repo](https://github.com/BeierZhu/GAN-MNIST-Pytorch/blob/master/main.py) (trained for 300 epochs).

I've also included a pre-trained LeNet classifier which achieves 99% test accuracy in the [mnist_classifier](mnist_classifier) folder, based on [this repo](https://github.com/activatedgeek/LeNet-5).

### cifar10

The cifar10 gan is from the [pytorch examples repo](https://github.com/pytorch/examples/tree/master/dcgan) and implements the [DCGAN paper](http://arxiv.org/abs/1511.06434). It required only minor alterations to generate images the size of the cifar10 dataset (32x32x3). Trained for 200 epochs. Weights [here](https://github.com/csinva/pytorch_gan_pretrained/tree/master/cifar10_dcgan/weights).

| data samples                                            |    generated samples                                        |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| ![real_images](cifar10_dcgan/samples/real_samples.png) | ![fake_images-300](cifar10_dcgan/samples/fake_samples_epoch_199.png)  |

I've also linked to a pre-trained cifar10 classifier in the [cifar10_classifier](cifar10_classifier) folder from [this repo](https://github.com/aaron-xichen/pytorch-playground/tree/master/cifar).



### cifar100

Similiar to the above gans, the cifar100 gan here generates 32x32x1 images for generating grayscale images. Trained for 200 epochs. Weights [here](https://github.com/csinva/pytorch_gan_pretrained/tree/master/cifar100_dcgan_grayscale/weights). There are also weights/code for generating images which are 34x45x1.

| data samples                                             |   generated samples                                        |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| ![real_images](cifar100_dcgan_grayscale/samples/real_samples.png) | ![fake_images-300](cifar100_dcgan_grayscale/samples/fake_samples_epoch_299.png) |

### reference
- based on the official [pytorch examples repo](https://github.com/pytorch/examples/tree/master/dcgan) with modifications to generate the appropriate size
- feel free to use/share this code openly
- for similar projects, see related repos: (e.g. [imodels](https://github.com/csinva/imodels), [neural-network-interpretations](https://github.com/csinva/acd)) or my website ([csinva.io](https://csinva.github.io/))
- tweets [@csinva_](https://twitter.com/csinva_)

