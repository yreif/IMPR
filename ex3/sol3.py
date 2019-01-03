# image processing ex3
# author: yuval.reif

import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d as conv2d
import matplotlib.pyplot as plt
import os
from imageio import imwrite


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    '''
    reads an image file and converts it into a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: image is represented by a matrix of type np.float64 with intensities (either grayscale or RGB channel
    intensities) normalized to the range [0, 1].
    '''
    im = imread(relpath(filename)).astype(np.float64) / 255
    if representation == 1:
        return rgb2gray(im)
    elif representation == 2:
        return im


def display_image(im, representation, subplot=None, title=None):
    '''
    displays an image in a given representation.
    :param im: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    '''
    if subplot is None:
        plt.figure()
    else:
        plt.subplot(subplot)
    if title is not None:
        plt.title(title)
    if representation == 1:
        plt.imshow(im, cmap="gray")
    if representation == 2:
        plt.imshow(im)
    plt.axis('off')
    if subplot is None:
        plt.show()


def build_gaussian_filter_vec(filter_size):
    '''
    computes the gaussian kernel of shape (kernel_size, kernel_size)
    :param filter_size: the size of the gaussian kernel in each dimension (an odd integer).
    :return: float64 array of shape (kernel_size, kernel_size)
    '''
    if filter_size == 1:
        kernel = np.ones((1, 1), dtype=np.float64)
    else:
        kernel = base_kernel = 0.25 * np.array([[1, 2, 1]], dtype=np.float64)
        for i in range(filter_size//2 - 1):
            kernel = conv2d(kernel, base_kernel)
    return kernel


def pyramid_reduce(im, filter_vec):
    reduced_im = convolve(im, filter_vec)
    reduced_im = convolve(reduced_im, filter_vec.T)
    return reduced_im[::2, ::2]


def pyramid_expand(im, filter_vec):
    expanded_im = np.zeros((2*im.shape[0], 2*im.shape[1]))
    expanded_im[::2, ::2] = im
    expanded_im = convolve(expanded_im, 2*filter_vec)
    expanded_im = convolve(expanded_im, 2*filter_vec.T)
    return expanded_im


def compute_n_levels(shape, max_levels):
    n_levels = max_levels
    while True:
        max_downscale_factor = 2**(n_levels-1)
        if shape[0]/max_downscale_factor >= 16 and shape[1]/max_downscale_factor >= 16:
            break
        n_levels -= 1
    return n_levels


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    constructs a Gaussian pyramid of a given image.
    :param im: a grayscale image with float values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
           to be used in constructing the pyramid filter.
    :return: a tuple (pyr, filter_vec) where:
             pyr - the resulting pyramid as a standard python array with maximum length of max_levels,
             where each element of the array is a grayscale image.
             filter_vec - a row vector of shape (1, filter_size) used for the pyramid construction.
    '''
    filter_vec = build_gaussian_filter_vec(filter_size)
    pyr = [im]
    n_levels = compute_n_levels(im.shape, max_levels)
    for level in range(n_levels-1):
        im = pyramid_reduce(im, filter_vec)
        pyr.append(im)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    constructs a Gaussian pyramid of a given image.
    :param im: a grayscale image with float values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
           to be used in constructing the pyramid filter.
    :return: a tuple (pyr, filter_vec) where:
             pyr - the resulting pyramid as a standard python array with maximum length of max_levels,
             where each element of the array is a grayscale image.
             filter_vec - a row vector of shape (1, filter_size) used for the pyramid construction.
    '''
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = list()
    for level in range(len(gaussian_pyr)-1):
        pyr.append(gaussian_pyr[level] - pyramid_expand(gaussian_pyr[level+1], filter_vec))
    pyr.append(gaussian_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    reconstructs an image from its laplacian pyramid
    :param lpyr: laplacian pyramid generated by build_laplacian_pyramid
    :param filter_vec: a row vector of shape (1, filter_size) used for the pyramid construction.
    :param coeff: list of the same length as the number of levels in the pyramid lpyr,
           used for reconstructing the image by multiplying each level i of lpyr by
           its corresponding coefficient
    :return: image reconstructed from its laplacian pyramid
    '''
    lpyr = [coeff[i] * lpyr[i] for i in range(len(lpyr))]
    im = lpyr.pop()
    while lpyr:
        im = pyramid_expand(im, filter_vec)
        im += lpyr.pop()
    return im


def stretch(im):
    im_min = im.min()
    im_max = im.max()
    return (im - im_min) / (im_max - im_min)


def render_pyramid(pyr, levels):
    '''
    builds a render of pyramid levels.
    :param pyr: a gaussian or a laplacian pyramid.
    :param levels: the number of levels to present in the result ≤ max_levels.
    :return: a single black image in which the pyramid levels of the given
             pyramid are stacked horizontally (after stretching the values to [0, 1]).
    '''
    rows = pyr[0].shape[0]
    levels = min(levels, len(pyr))
    cols = sum([level.shape[1] for level in pyr[:levels]])
    result = np.zeros((rows, cols))
    curr_col = 0
    for i in range(levels):
        level = stretch(pyr[i])
        rows, cols = level.shape
        result[:rows, curr_col:curr_col+cols] = level
        curr_col += cols
    return result


def display_pyramid(pyr, levels):
    '''
    displays a render of pyramid levels.
    :param pyr: a gaussian or a laplacian pyramid.
    :param levels: the number of levels to present in the result ≤ max_levels.
    :return: a single black image in which the pyramid levels of the given
             pyramid are stacked horizontally (after stretching the values to [0, 1]).
    '''
    display_image(render_pyramid(pyr, levels), 1)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    pyramid blends the two images according to the mask.
    im1, im2 and mask should all have the same dimensions that are multiples of 2**(max_levels−1) .
    :param im1: input grayscale image to be blended.
    :param im2: input grayscale image to be blended.
    :param mask: a boolean (i.e. dtype == np.bool) mask containing True and False representing
           which parts of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: the max_levels parameter to use when generating the Gaussian and Laplacian
           pyramids.
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared
           filter) which defines the filter used in the construction of the Laplacian pyramids
           of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter (an odd scalar that represents a squared
           filter) which defines the filter used in the construction of the Gaussian pyramid of mask.
    :return:
    '''
    l_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l_2, __ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    g_m, __ = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = [g_m[k]*l_1[k] + (1-g_m[k])*l_2[k] for k in range(max_levels)]
    return laplacian_to_image(l_out, filter_vec, [1] * len(l_out)).clip(0, 1)


def pyramid_blending_rpg(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    pyramid blends the two images according to the mask.
    im1, im2 and mask should all have the same dimensions that are multiples of 2**(max_levels−1) .
    :param im1: input RGB image to be blended.
    :param im2: input RGB image to be blended.
    :param mask: a boolean (i.e. dtype == np.bool) mask containing True and False representing
           which parts of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: the max_levels parameter to use when generating the Gaussian and Laplacian
           pyramids.
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared
           filter) which defines the filter used in the construction of the Laplacian pyramids
           of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter (an odd scalar that represents a squared
           filter) which defines the filter used in the construction of the Gaussian pyramid of mask.
    :return:
    '''
    result = np.zeros(im1.shape)
    for i in range(im1.shape[2]):
        result[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels,
                                           filter_size_im, filter_size_mask)
    return result


def blending_example1():
    im1 = read_image('externals/example1/kimono.jpg', 2)
    im2 = read_image('externals/example1/dog.jpg', 2)
    mask = read_image('externals/example1/mask.jpg', 1)
    mask[mask < 0.5] = 0
    mask = mask.astype(np.bool)
    
    im_blend = pyramid_blending_rpg(im1, im2, mask, max_levels=7, filter_size_im=11, filter_size_mask=11)
    display_image(im1, 2, 221)
    display_image(im2, 2, 222)
    display_image(mask, 1, 223)
    display_image(im_blend, 2, 224)
    plt.show()
    return im1, im2, mask, im_blend

