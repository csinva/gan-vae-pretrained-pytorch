**Pre-trained GANs for MNIST / CIFAR10**

- includes model class definitions
- includes training scripts
- includes notebooks showing how to load pretrained nets / use them
- tested with pytorch 1.0, python 3.5

### mnist

mnist code from [this repo](https://github.com/BeierZhu/GAN-MNIST-Pytorch/blob/master/main.py). Trained for 300 epochs. Weights [here](mnist/weights).



| generated samples | data samples |
| ----------------- | ------------ |
| ![fake_images-300](mnist/samples/fake_images-300.png) | ![real_images](mnist/samples/real_images.png) |



### cifar10

The cifar10 gan is from the [pytorch examples repo](https://github.com/pytorch/examples/tree/master/dcgan) and implements the [DCGAN paper](http://arxiv.org/abs/1511.06434). It required only minor alterations to generate images the size of the cifar10 dataset (32x32) and was trained for 200 epochs. Weights [here](cifar10/weights).

| generated samples                                            | data samples                                     |
| ------------------------------------------------------------ | ------------------------------------------------ |
| ![fake_images-300](cifar10/samples/fake_samples_epoch_199.png) | ![real_images](cifar10/samples/real_samples.png) |
