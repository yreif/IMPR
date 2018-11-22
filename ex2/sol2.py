# image processing ex1
# author: yuval.reif

import numpy as np
from numpy.fft import fftshift, ifftshift
from scipy.misc import imread
from scipy.signal import convolve2d as conv2d
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def read_image(filename, representation):
    '''
    reads an image file and converts it into a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: image is represented by a matrix of type np.float64 with intensities (either grayscale or RGB channel
    intensities) normalized to the range [0, 1].
    '''
    im = imread(filename).astype(np.float64)/255
    if representation == 1:
        return rgb2gray(im)
    elif representation == 2:
        return im


def __display_image(im, representation, title):
    '''
    displays an image in a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    '''
    plt.figure()
    plt.title(title)
    if representation == 1:
        plt.imshow(im, cmap="gray")
    if representation == 2:
        plt.imshow(im)
    plt.axis('off')
    plt.show()


def DFT(signal):
    '''
    transforms a 1D discrete signal/a matrix which columns are 1D discrete signals to its Fourier representation.
    :param signal: an array of dtype float64 with shape (N,M) (or complex128 if M>1)
    :return: an array of dtype complex128 with shape (N,M).
    '''
    n = signal.shape[0]
    coefficients = np.exp(-2j * np.pi * np.arange(n) / n).reshape(-1, 1)
    coefficients = coefficients ** np.arange(n)
    return np.matmul(coefficients, signal)


def IDFT(fourier_signal):
    '''
    transforms a Fourier representation of a 1D discrete signal or a matrix which columns are 1D discrete signals back
    to the original signal.
    :param fourier_signal: an array of dtype complex128 with shape (N,M).
    :return: an array of dtype complex128 with shape (N,M).
    '''
    n, m = fourier_signal.shape
    coefficients = np.exp(2j * np.pi * np.arange(n) / n).reshape(-1, 1)
    coefficients = coefficients ** np.arange(n)
    return np.matmul(coefficients, fourier_signal) / n


def DFT2(image):
    '''
    converts a 2D discrete signal to its Fourier representation.
    :param image: a grayscale image of dtype float64.
    :return: an array of dtype complex128 with the same shape as image.
    '''
    return DFT(DFT(image.T).T)


def IDFT2(fourier_image):
    '''
    converts a 2D discrete Fourier representations to a signal.
    :param fourier_image: a 2D array of dtype complex128.
    :return: a 2D array of dtype complex128 with the same shape as fourier_image.
    '''
    return IDFT(IDFT(fourier_image.T).T)


def conv_der(im):
    '''
    computes the magnitude of image derivatives.
    :param im: grayscale image of dtype float64.
    :return: grayscale image of dtype float64 which is is the magnitude of the derivative, with the same
             shape as the original image.
    '''
    dx = conv2d(im, np.array([1, 0, -1]).reshape(1, -1), mode='same')
    dy = conv2d(im, np.array([1, 0, -1]).reshape(-1, 1), mode='same')
    return np.sqrt(dx**2 + dy**2)


def fourier_der(im):
    '''
    computes the magnitude of image derivatives using Fourier transform.
    :param im: grayscale image of dtype float64.
    :return: grayscale image of dtype float64 which is is the magnitude of the derivative, with the same
             shape as the original image.
    '''
    n, m = im.shape
    fourier_im = fftshift(DFT2(im))
    dx = (2j/n * np.pi) * IDFT2(ifftshift(fourier_im * np.arange(-n//2, n//2).reshape(-1, 1)))
    dy = (2j/m * np.pi) * IDFT2(ifftshift(fourier_im * np.arange(-m//2, m//2).reshape(1, -1)))
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def __gaussian_kernel(kernel_size):
    '''
    computes the gaussian kernel of shape (kernel_size, kernel_size)
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer).
    :return: float64 array of shape (kernel_size, kernel_size)
    '''
    if kernel_size == 1:
        kernel = np.ones((1, 1), dtype=np.float64)
    else:
        kernel = base_kernel = 0.25 * np.array([[1, 2, 1]], dtype=np.float64)
        for i in range(kernel_size//2 - 1):
            kernel = conv2d(kernel, base_kernel)
        kernel = conv2d(kernel, kernel.T)
    return kernel


def blur_spatial(im, kernel_size):
    '''
    performs image blurring using 2D convolution between the image and a gaussian kernel.
    :param im: a grayscale float64 image to be blurred.
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer).
    :return: output grayscale float64 blurry image.
    '''
    return conv2d(im, __gaussian_kernel(kernel_size), mode='same', boundary='symm')


def blur_fourier(im, kernel_size):
    '''
    performs image blurring in fourier space with a gaussian kernel.
    :param im: a grayscale float64 image to be blurred.
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer).
    :return: output grayscale float64 blurry image.
    '''
    m, n = im.shape
    kernel = np.zeros((m, n))
    kernel_location = (slice(m//2 - kernel_size//2, m//2 + kernel_size//2 + 1),
                       slice(n//2 - kernel_size//2, n//2 + kernel_size//2 + 1))
    kernel[kernel_location[0], kernel_location[1]] = __gaussian_kernel(kernel_size)
    fourier_kernel = fftshift(DFT2(ifftshift(kernel)))
    fourier_im = fftshift(DFT2(im))
    return np.real(IDFT2(ifftshift(fourier_im * fourier_kernel)))



