# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
from scipy.ndimage.filters import convolve
from scipy.misc import imsave
from numpy.linalg import inv

import sol4_utils


DESC_RAD = 3
SPREADOUT_N = 7
SPREADOUT_M = 7


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    blur_kernel_size = 3
    k = 0.04
    dx_filter = np.array([[1, 0, -1]])

    Ix = convolve(im, dx_filter)
    Iy = convolve(im, dx_filter.T)
    Ix_square = sol4_utils.blur_spatial(Ix ** 2, blur_kernel_size)
    Iy_square = sol4_utils.blur_spatial(Iy ** 2, blur_kernel_size)
    IxIy = sol4_utils.blur_spatial(Ix * Iy, blur_kernel_size)
    M = np.zeros((im.shape[0], im.shape[1], 2, 2))
    M[:, :, 0, 0] = Ix_square
    M[:, :, 0, 1] = IxIy
    M[:, :, 1, 0] = IxIy
    M[:, :, 1, 1] = Iy_square
    R = np.linalg.det(M) - k * (np.trace(M, axis1=2, axis2=3) ** 2)

    max_xy = np.flip(np.column_stack(np.nonzero(non_maximum_suppression(R))), axis=1)
    return max_xy


def sample_descriptor(im, pos, desc_rad=3):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    n = pos.shape[0]
    k = 2*desc_rad + 1
    desc = np.zeros((n, k, k))
    rad_cords = np.meshgrid(np.arange(-desc_rad, desc_rad + 1), np.arange(-desc_rad, desc_rad + 1))
    for i in range(n):
        patch_cords = (rad_cords[1] + pos[i, 1], rad_cords[0] + pos[i, 0])
        desc[i] = map_coordinates(im, patch_cords, order=1, prefilter=False)
    desc = desc - np.mean(desc, axis=(1, 2)).reshape(-1, 1, 1)
    desc_norm = np.linalg.norm(desc, axis=(1, 2))
    desc_norm[desc_norm == 0] = np.inf
    desc = desc/desc_norm.reshape(-1, 1, 1)

    return desc


def levels_translation(points, orig_level, dest_level):
    '''
    translates points array from original level in the pyramid to the destination level.
    '''
    return 2**(orig_level - dest_level) * points


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    desc_pyr_level = 2
    corners = spread_out_corners(pyr[0], SPREADOUT_M, SPREADOUT_N, 2*DESC_RAD + 1)
    pos = levels_translation(corners, 0, desc_pyr_level)
    desc = sample_descriptor(pyr[desc_pyr_level], pos, DESC_RAD)
    return [corners, desc]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    n1 = len(desc1)
    n2 = len(desc2)
    # score[i, j] is the match score of desc1[i], desc2[j]
    score = desc1.reshape(n1, -1) @ desc2.reshape(n2, -1).T
    matches = np.zeros(score.shape)
    best_2_desc1 = np.argpartition(score, -2, axis=1)[:, -2:]
    best_2_desc2 = np.argpartition(score, -2, axis=0)[-2:, :]

    matches[np.arange(n1).reshape(-1, 1), best_2_desc1] += 1
    matches[best_2_desc2, np.arange(n2)] += 1
    return np.nonzero(np.logical_and(score > min_score, matches == 2))


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    orig = np.ones((len(pos1), 3, 1))
    orig[:, :2, 0] = pos1
    pos2 = H12 @ orig
    pos2 = pos2[:, :2, 0] / pos2[:, 2, :]
    return pos2


def ransac_homography(pos1, pos2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    N = len(pos1)
    best_inliners_n = 0
    best_inliners_ind = 0
    for _ in range(num_iter):
        if translation_only:
            points_ind = np.random.choice(N, size=1)
        else:
            points_ind = np.random.choice(N, size=2)
        h = estimate_rigid_transform(pos1[points_ind], pos2[points_ind], translation_only)
        h_pos1 = apply_homography(pos1, h)
        dist = np.linalg.norm(h_pos1 - pos2, axis=1)**2
        inliers_ind = np.nonzero(dist < inlier_tol)[0]
        inliers_n = len(inliers_ind)
        if inliers_n > best_inliners_n:
            best_inliners_n = inliers_n
            best_inliners_ind = inliers_ind
    best_h = estimate_rigid_transform(pos1[best_inliners_ind], pos2[best_inliners_ind], translation_only)
    return [best_h, best_inliners_ind]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2 An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), cmap='gray')
    im1_x = len(im1[0])
    n = len(points1)
    for i in range(n):
        if i in inliers:
            plt.plot([points1[i, 0], points2[i, 0] + im1_x],
                     [points1[i, 1], points2[i, 1]],
                     c='y', mfc='r', lw=.4, ms=3, marker='o')
        else:
            plt.plot([points1[i, 0], points2[i, 0] + im1_x],
                     [points1[i, 1], points2[i, 1]],
                     c='b', mfc='r', lw=.4, ms=3, marker='o')

    plt.show()


def accumulate_homographies(H_successive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    n = len(H_successive)
    H2m = [np.eye(3)]
    curr_h = H2m[0]
    for i in range(m, 0, -1):
        curr_h = curr_h @ H_successive[i - 1]
        H2m.append(curr_h / curr_h[2, 2])
    H2m.reverse()
    curr_h = H2m[-1]
    for i in range(m, n):
        curr_h = curr_h @ inv(H_successive[i])
        H2m.append(curr_h / curr_h[2, 2])
    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = [[0, 0], [w, 0], [0, h], [w, h]]
    warped_corners = apply_homography(corners, homography)

    top_left_corner = [warped_corners[:, 0].min(), warped_corners[:, 1].min()]
    bottom_right_corner = [warped_corners[:, 0].max(), warped_corners[:, 1].max()]
    return np.array([top_left_corner, bottom_right_corner]).astype(int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    top_left_corner, bottom_right_corner = compute_bounding_box(homography, image.shape[1], image.shape[0])
    warped_coords = np.meshgrid(np.arange(top_left_corner[0], bottom_right_corner[0]),
                                 np.arange(top_left_corner[1], bottom_right_corner[1]))
    warped_coords = np.array(warped_coords)
    coords_shape = warped_coords.shape
    orig_coords = apply_homography(warped_coords.T.reshape(-1, 2), inv(homography)).reshape(coords_shape[::-1]).T
    warped_im = map_coordinates(image, np.flip(orig_coords, axis=0), order=1, prefilter=False)

    return warped_im


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


def crop(im, new_x, new_y):
    '''
    center-crops im to desired size.
    '''
    if len(im.shape) == 3:
        y, x, c = im.shape
        start_x = (x//2) - (new_x//2)
        start_y = (y//2) - (new_y//2)
        return im[start_y:start_y+new_y, start_x:start_x+new_x, :]
    else:
        y, x = im.shape
        start_x = (x//2) - (new_x//2)
        start_y = (y//2) - (new_y//2)
        return im[start_y:start_y+new_y, start_x:start_x+new_x]


# I didn't know if this function would be in the school's sol4_utils so I copied it here.
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
        result[:, :, i] = sol4_utils.pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels,
                                           filter_size_im, filter_size_mask)
    return result


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        # self.images = []
        self.bonus = bonus
        self.blending_levels = 6
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            # self.images.append(image)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        # print('range:', len(points_and_descriptors) - 1)
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]
            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]


            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # if i < 5:
            #     display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        if self.bonus:
            # 1. create two panoramas for each i in number_of_panoramas:
            #   one from even-indexed frames & the other from odd-indexed frames,
            #   taking two image strips from each frame instead of one (except for the 1-indexed frame,
            #   from which we take only one image strip so that the the odd and even panoramas
            #   will have the stitches between image strips in different locations).
            # 2. for each i, blend the two together with a barcode mask.
            barcode_panoramas = np.zeros((number_of_panoramas, 2, panorama_size[1], panorama_size[0], 3),
                                         dtype=np.float64)
            barcode_mask = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0]))
            min_strip_size = 1000
            for i, frame_index in enumerate(self.frames_for_panoramas):
                # warp every input image once, and populate all panoramas
                image = sol4_utils.read_image(self.files[frame_index], 2)
                warped_image = warp_image(image, self.homographies[i])
                x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
                y_bottom = y_offset + warped_image.shape[0]

                if i == 1:  # we take only one image strip
                    for panorama_index in range(number_of_panoramas):
                        # take strip of warped image and paste to current panorama
                        boundaries = x_strip_boundary[panorama_index, i:i + 2]
                        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                        x_end = boundaries[0] + image_strip.shape[1]
                        barcode_panoramas[panorama_index, 1, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

                elif i == len(self.frames_for_panoramas) - 1:  # we take only one strip because this is the last frame
                    for panorama_index in range(number_of_panoramas):
                        # take strip of warped image and paste to current panorama
                        boundaries = x_strip_boundary[panorama_index, i:i + 2]
                        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                        x_end = boundaries[0] + image_strip.shape[1]
                        barcode_panoramas[panorama_index, i % 2, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

                else:  # we take two image strips. if i is even, we also update the barcode mask.
                    for panorama_index in range(number_of_panoramas):
                        if i % 2 == 0:
                            # take strip of warped image and paste to current panorama
                            boundaries = x_strip_boundary[panorama_index, i:i + 3][[0, 2]]
                            image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                            x_end = boundaries[0] + image_strip.shape[1]
                            barcode_panoramas[panorama_index, 0, y_offset:y_bottom, boundaries[0]:x_end] = image_strip
                            mask_boundaries = x_strip_boundary[panorama_index, i + 1:i + 3]
                            barcode_mask[panorama_index, :, mask_boundaries[0]:x_end] = 1
                        else:
                            # take strip of warped image and paste to current panorama
                            boundaries = x_strip_boundary[panorama_index, i-2:i + 1][[0, 2]]
                            image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                            x_end = boundaries[0] + image_strip.shape[1]
                            barcode_panoramas[panorama_index, 1, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

                        if image_strip.shape[1] < min_strip_size:
                            min_strip_size = image_strip.shape[1]
            # barcode-blend panoramas
            factor = 2**(self.blending_levels-1)
            cropped_size = (panorama_size // factor) * factor
            # choose filter size for mask
            mask_filter_size = min_strip_size//2
            if mask_filter_size % 2 == 0:
                mask_filter_size -= 1
            if mask_filter_size == 1:
                mask_filter_size = 3
            self.panoramas = np.zeros((number_of_panoramas, cropped_size[1], cropped_size[0], 3), dtype=np.float64)
            for panorama_index in range(number_of_panoramas):
                if np.array_equal(panorama_size, cropped_size):  # no need to crop
                    im1 = barcode_panoramas[panorama_index, 0]
                    im2 = barcode_panoramas[panorama_index, 1]
                    mask = barcode_mask[panorama_index]
                else:  # crop to fit desired number of levels for pyramid blending
                    im1 = crop(barcode_panoramas[panorama_index, 0], cropped_size[0], cropped_size[1])
                    im2 = crop(barcode_panoramas[panorama_index, 1], cropped_size[0], cropped_size[1])
                    mask = crop(barcode_mask[panorama_index], cropped_size[0], cropped_size[1])
                self.panoramas[panorama_index] = pyramid_blending_rpg(im2, im1, mask,
                                                                      max_levels=self.blending_levels,
                                                                      filter_size_im=13,
                                                                      filter_size_mask=mask_filter_size)
        else:  # none-bonus implementation
            self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
            for i, frame_index in enumerate(self.frames_for_panoramas):
                # warp every input image once, and populate all panoramas
                image = sol4_utils.read_image(self.files[frame_index], 2)
                warped_image = warp_image(image, self.homographies[i])
                x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
                y_bottom = y_offset + warped_image.shape[0]

                for panorama_index in range(number_of_panoramas):
                    # take strip of warped image and paste to current panorama
                    boundaries = x_strip_boundary[panorama_index, i:i + 2]
                    image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                    x_end = boundaries[0] + image_strip.shape[1]
                    self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


