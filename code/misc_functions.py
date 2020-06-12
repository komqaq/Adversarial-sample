import copy
import cv2
import numpy as np
import torchvision.transforms as trans
import torch
from torch.autograd import Variable
from torchvision import models
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
def preprocess_image(cv2im,source='imagenet'):
    # mean and std list for channels (Imagenet)
    if source == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif source == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif source == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var,source='imagenet'):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    if source == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif source == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif source == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] *= std[c]
        recreated_im[c] += mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


def get_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        pretrained_model(Pytorch model): Model to use for the operations
    """

    img_path = sys.path[0]+'/../input_images/'+ str(example_index) +'.jpg'
    # Read image
    original_image = cv2.imread(img_path, 1)
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    
    return (original_image,
            prep_img)
