# image processing ex1
# author: yuval.reif

import numpy as np
from math import ceil
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# CONSTANTS:
RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ2RGB = np.linalg.inv(RGB2YIQ)


def __is_monotone(seq, seq_name='sequence', direction='strictly increasing', check_upto=0):
    if check_upto == 0:
        check_upto = len(seq)
    if direction == 'decreasing':
        for i in range(1, check_upto):
            if seq[i] > seq[i-1]:
                print(seq_name + ' is not monotonically decreasing')
                return False
        return True
    else:  # strictly increasing
        for i in range(1, check_upto):
            if seq[i] <= seq[i-1]:
                print(seq_name + ' is not strictly monotonically increasing')
                return False
        return True


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


def imdisplay(filename, representation):
    '''
    displays an image in a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    '''
    im = read_image(filename, representation)
    if im is not None:
        plt.figure()
        if representation == 1:
            plt.imshow(im, cmap="gray")
        if representation == 2:
            plt.imshow(im)
        plt.axis('off')
        plt.show()


def rgb2yiq(imRGB):
    '''
    transforms an RGB image into the YIQ color space.
    :param imRGB: RGB image represented by a height × width × 3 np.float64 matrice with values in [0, 1].
    :return:
    '''
    return np.dot(imRGB, RGB2YIQ.transpose())


def yiq2rgb(imYIQ):
    '''
    transforms an RGB image into the YIQ color space.
    :param imRGB: RGB image represented by a height × width × 3 np.float64 matrice with values in [0, 1].
    :return:
    '''
    return np.dot(imYIQ, YIQ2RGB.transpose())


def histogram_equalize(im_orig):
    '''
    performs histogram equalization of a given grayscale or RGB image
    :param im_orig: input grayscale or RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
    - im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
    - hist_orig - is a 256 bin histogram of the original image (array with shape (256,)).
    - hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,)).
    '''
    if len(im_orig.shape) == 2:  # grayscale case
        im = np.around(im_orig * 255).astype(np.uint8)
        hist = np.histogram(im, bins=256, range=(0, 255))[0]
        cum_hist = np.cumsum(hist)
        m = np.argmax(cum_hist != 0)  # Let m be first grey level for which cum_hist(m) != 0
        intensity_map = np.round((cum_hist - cum_hist[m]) / (cum_hist[255] - cum_hist[m]) * 255).astype(np.uint8)
        im_eq = np.reshape(np.clip(intensity_map[im.flatten()]/255, 0, 1), (im.shape[0], im.shape[1]))
        hist_eq = np.histogram(im_eq, 256)[0]
        return [im_eq, hist, hist_eq]

    elif len(im_orig.shape) == 3:  # RGB case
        imYIQ = rgb2yiq(im_orig)
        Y_orig = imYIQ[:, :, 0].copy()
        Y_eq, hist, hist_eq = histogram_equalize(Y_orig)
        imYIQ[:, :, 0] = Y_eq
        im_eq = yiq2rgb(imYIQ)
        return [im_eq, hist, hist_eq]


def __create_quantization_map(n_quant, q, z):
    map = np.zeros(256)
    for i in range(n_quant):
        map[z[i]:(z[i+1] + 1)] = q[i]
    return map


def quantize(im_orig, n_quant, n_iter):
    '''
    performs optimal quantization of a given grayscale or RGB image (if RGB, quantization is performed only
    on the luminance channel of the image).
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of intensities the output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure (may converge earlier).
    :return: a list [im_quant, error] where
            - im_quant - is the quantized output image.
            - error - is an array with shape (n_iter,) (or less) of the total intensities error for each
              iteration of the quantization procedure.
    '''

    error = np.zeros(n_iter)
    q = np.zeros(n_quant)

    if len(im_orig.shape) == 2:  # greyscale case
        w, h = tuple(im_orig.shape)
        im = np.around(im_orig * 255).astype(np.uint8)
        hist = np.histogram(im, 256, range=(0, 255))[0]
        cum_pixels = np.cumsum(hist)
        cum_weighted_pixels = np.cumsum(hist * np.arange(256))
        for j in range(n_iter):
            if j == 0:  # initialize z values:
                z = np.sort(im.flatten())[0::ceil(w * h / n_quant)]
                z[0] = 0
                z = np.append(z, 255)
            else:  # update z values:
                # if the new z isn't an integer, then the fraction is necessarily 0.5,
                # so we round it in a way that would lead to lowest quantization error
                prev_z = z.copy()
                for i in range(1, n_quant):
                    new_z_val = ((q[i-1] + q[i])/2)
                    if z[i] == new_z_val:
                        pass
                    elif int(new_z_val) == new_z_val:
                        z[i] = new_z_val
                    else:  # round z
                        new_z_floor = int(new_z_val)
                        if (new_z_floor - q[i-1]) > (q[i] - new_z_floor):
                            z[i] = new_z_floor
                        else:
                            z[i] = new_z_floor + 1
            # update q values
            q[0] = (cum_weighted_pixels[z[1] - 1] - cum_weighted_pixels[z[0]]) / cum_pixels[z[1] - 1]
            for i in range(1, n_quant - 1):
                q[i] = (cum_weighted_pixels[z[i + 1] - 1] - cum_weighted_pixels[z[i] - 1]) / \
                       (cum_pixels[z[i + 1] - 1] - cum_pixels[z[i] - 1])
            q[n_quant - 1] = (cum_weighted_pixels[z[n_quant]] - cum_weighted_pixels[z[n_quant - 1] - 1]) / \
                             (cum_pixels[z[n_quant]] - cum_pixels[z[n_quant - 1] - 1])

            q = np.around(q)
            # calculate error: \sum{i=0...k}\sum{z}(q_i - z)^2 * p(z)
            error[j] = np.sum((__create_quantization_map(n_quant, q, z) - np.arange(256))**2 * hist)
            if j > 0 and np.array_equal(z, prev_z):  # reached local minimum
                error = error[:j+1]
                break
        # create im_quant:
        intensity_map = __create_quantization_map(n_quant, q, z)
        im_quant = np.reshape(intensity_map[im.flatten()]/255, (w, h))
        return [im_quant, error]

    elif len(im_orig.shape) == 3:  # RGB case
        imYIQ = rgb2yiq(im_orig)
        Y_orig = imYIQ[:, :, 0].copy()
        Y_quant, error = quantize(Y_orig, n_quant, n_iter)
        imYIQ[:, :, 0] = Y_quant
        im_quant = yiq2rgb(imYIQ)
        return [im_quant, error]


def quantize_rgb(im_orig, n_quant):
    '''
    performs quantization for full color images.
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of intensities the output im_quant image should have.
    :return: the quantized output image.
    '''
    if len(im_orig.shape) == 3:  # RGB case

        # I tried a few versions -
        # - best of N random color choices: faster with ok-to-good results for high n_quant values,
        #   bad results for smaller n_quant values.
        # - k means on full image: error is always a local minimum, but slower for higher n_quant values
        # - k means on a sample of pixels: results almost as good as the previous version, but much faster
        #   for higher n_quant values
        # I decided to use both versions of k means - full on n_quant<=12, and partial otherwise.

        w, h, d = tuple(im_orig.shape)
        im_flattened = im_orig.reshape(w * h, d)

        if n_quant <= 12:
            # full image k means version:
            kmeans = KMeans(n_clusters=n_quant, n_init=5)
            labels = kmeans.fit_predict(im_flattened)
            colors = kmeans.cluster_centers_
        else:
            # partial image k means version:
            kmeans = KMeans(n_clusters=n_quant, n_init=5)
            indices = np.arange(w * h)
            kmeans.fit(im_flattened[np.random.choice(indices, size=ceil(w*h/8))])
            labels = kmeans.predict(im_flattened)
            colors = kmeans.cluster_centers_

        # # best of N random choices version:
        # N = 50  # constant for number of iterations
        # indices = np.arange(w*h)
        # curr_error = 0
        # for i in range(N):
        #     prev_error = curr_error
        #     colors = im_flattened[np.random.choice(indices, size=n_quant)]
        #     labels, cur_error = pairwise_distances_argmin_min(im_flattened, colors)
        #     curr_error = np.sum(curr_error)
        #     if i == 0:
        #         best_colors = colors
        #         best_labels = labels
        #         continue
        #     if curr_error < prev_error:
        #         best_colors = colors
        #         best_labels = labels
        # colors = best_colors
        # labels = best_labels

        im_quant = np.reshape(colors[labels], im_orig.shape)
        return im_quant
