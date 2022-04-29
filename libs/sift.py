import cv2 as cv
import numpy as np
import os
import logging
from functools import cmp_to_key
class SIFT:
    def __init__(self, image_path,pre_segma = 0.5) -> None:
        self.float_tolerance = 1e-7
        self.image_path = image_path
        self.pre_segma = pre_segma

    def computeKeypointsAndDescriptors(self,sigma=1.6, num_intervals=3, image_border_width=5):
        image = cv.imread(self.image_path,flags = cv.IMREAD_GRAYSCALE)
        image = image.astype('float32')
        # base_image = generateBaseImage(image, sigma, assumed_blur)
        base_image = self.generate_base_image(image, sigma, self.pre_segma)
        # num_octaves = computeNumberOfOctaves(base_image.shape)
        num_octaves = self.compute_number_of_octaves(base_image.shape)
        # gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
        gaussian_kernels = self.generate_gaussian_segmas(sigma, num_intervals)
        # gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
        gaussian_images = self.generate_gaussian_images(base_image, num_octaves, gaussian_kernels)
        # dog_images = generateDoGImages(gaussian_images)
        dog_images = self.generate_DoG_images(gaussian_images)
        # keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        keypoints = self.findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        # keypoints = removeDuplicateKeypoints(keypoints)
        keypoints = self.remove_duplicate_key_points(keypoints)
        # keypoints = convertKeypointsToInputImageSize(keypoints)
        keypoints = self.convert_key_points_to_input_image_size(keypoints)
        # descriptors = generateDescriptors(keypoints, gaussian_images)
        descriptors = self.generate_descriptors(keypoints, gaussian_images)
        return keypoints,descriptors

    ################################ Constructing a Scale Space ###########################

    def generate_base_image(self, image, sigma, pre_segma):
        image = cv.resize(image, (0, 0), fx=2, fy=2,
                          interpolation=cv.INTER_LINEAR)
        sigma_diff = np.sqrt(
            max((sigma ** 2) - ((2 * pre_segma) ** 2), 0.01))
        return cv.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    def compute_number_of_octaves(self, image_shape):
        return int(np.round(np.log(min(image_shape)) / np.log(2) - 1))

    def generate_gaussian_segmas(self, sigma, num_intervals):
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_segmas = np.zeros(num_images_per_octave)
        gaussian_segmas[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_segmas[image_index] = np.sqrt(
                sigma_total ** 2 - sigma_previous ** 2)

        return gaussian_segmas

    def generate_gaussian_images(self, image, num_octaves, gaussian_segmas):
        gaussian_images = []
        for octave_index in range(num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(image)
            for gaussian_segma in gaussian_segmas[1:]:
                image = cv.GaussianBlur(
                    image, (0, 0), sigmaX=gaussian_segma, sigmaY=gaussian_segma)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = cv.resize(octave_base, (int(
                octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv.INTER_NEAREST)
        return np.array(gaussian_images, dtype=object)

    def generate_DoG_images(self, gaussian_images):
        dog_images = []
        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                # ordinary subtraction will not work because the images are unsigned integers
                dog_images_in_octave.append(
                    np.subtract(second_image, first_image))
            dog_images.append(dog_images_in_octave)
        return np.array(dog_images, dtype=object)

    ################################ Keypoint Localisation ###########################

    def findScaleSpaceExtrema(self, gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
        # from OpenCV implementation
        threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
        keypoints = []
        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                # (i, j) is the center of the 3x3 array
                for i in range(image_border_width, first_image.shape[0] - image_border_width):
                    for j in range(image_border_width, first_image.shape[1] - image_border_width):
                        if self.is_pixel_an_extremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                            localization_result = self.localize_extremum_via_quadratic_fit(
                                i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = self.compute_key_points_with_orientations(
                                    keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                                for keypoint_with_orientation in keypoints_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints

    def is_pixel_an_extremum(self, first_subimage, second_subimage, third_subimage, threshold):
        center_pixel_value = second_subimage[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_subimage) and \
                    np.all(center_pixel_value >= third_subimage) and \
                    np.all(center_pixel_value >= second_subimage[0, :]) and \
                    np.all(center_pixel_value >= second_subimage[2, :]) and \
                    center_pixel_value >= second_subimage[1, 0] and \
                    center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_subimage) and \
                    np.all(center_pixel_value <= third_subimage) and \
                    np.all(center_pixel_value <= second_subimage[0, :]) and \
                    np.all(center_pixel_value <= second_subimage[2, :]) and \
                    center_pixel_value <= second_subimage[1, 0] and \
                    center_pixel_value <= second_subimage[1, 2]
        return False

    def localize_extremum_via_quadratic_fit(self, i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            first_image, second_image, third_image = dog_images_in_octave[
                image_index-1:image_index+2]
            pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                                   second_image[i-1:i+2, j-1:j+2],
                                   third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.compute_gradient_at_center_pixel(pixel_cube)
            hessian = self.compute_hessian_at_center_pixel(pixel_cube)
            extremum_update = - \
                np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(np.round(extremum_update[0]))
            i += int(np.round(extremum_update[1]))
            image_index += int(np.round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1,
                                                    1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV KeyPoint object
                keypoint = cv.KeyPoint()
                keypoint.pt = (
                    (j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * \
                    (2 ** 8) + \
                    int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (
                    2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None

    def compute_gradient_at_center_pixel(self, pixel_array):
        # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
        # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def compute_hessian_at_center_pixel(self, pixel_array):
        # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
        # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
        # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
        # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * \
            center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * \
            center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * \
            center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1,
                      2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2,
                      1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2,
                      0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs],
                         [dxy, dyy, dys],
                         [dxs, dys, dss]])

    ################################ Orientation Assignment ###########################

    def compute_key_points_with_orientations(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):

        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
        scale = scale_factor * keypoint.size / \
            np.float32(2 ** (octave_index + 1))
        radius = int(np.round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(
                np.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(
                        np.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - \
                            gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - \
                            gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        # constant in front of exponential can be dropped because we will find peaks later
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_index = int(
                            np.round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index %
                                      num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(
                n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(
            smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                # Quadratic peak interpolation
                # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                    left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < self.float_tolerance:
                    orientation = 0
                new_keypoint = cv.KeyPoint(
                    *keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    ################################ Keypoint Descriptor ###########################
    def compare_key_points(self, keypoint1, keypoint2):
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def convert_key_points_to_input_image_size(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | (
                (keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def remove_duplicate_key_points(self, keypoints):

        if len(keypoints) < 2:
            return keypoints
        keypoints.sort(key=cmp_to_key(self.compare_key_points))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
                    last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
                    last_unique_keypoint.size != next_keypoint.size or \
                    last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)

        return unique_keypoints

    def unpack_octave(self, keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / \
            np.float32(1 << octave) if octave >= 0 else np.float32(
                1 << -octave)
        return octave, layer, scale

    def generate_descriptors(self, keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        descriptors = []
        for keypoint in keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            gaussian_image = gaussian_images[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            # first two dimensions are increased by 2 to account for border effects
            histogram_tensor = np.zeros(
                (window_width + 2, window_width + 2, num_bins))

            # Descriptor window size (described by half_width) follows OpenCV convention
            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            # sqrt(2) corresponds to diagonal length of a pixel
            half_width = int(np.round(hist_width * np.sqrt(2)
                             * (window_width + 1) * 0.5))
            # ensure half_width lies within image
            half_width = int(
                min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(np.round(point[1] + row))
                        window_col = int(np.round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - \
                                gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - \
                                gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(
                                np.arctan2(dy, dx)) % 360
                            weight = np.exp(
                                weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append(
                                (gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                # Smoothing via trilinear interpolation
                # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
                # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor(
                    [row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - \
                    row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1,
                                 col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor +
                                 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1,
                                 col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor +
                                 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2,
                                 col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor +
                                 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2,
                                 col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor +
                                 2, (orientation_bin_floor + 1) % num_bins] += c111

            # Remove histogram borders
            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            # Threshold and normalize descriptor_vector
            threshold = np.linalg.norm(
                descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_tolerance)
            # Multiply by 512, np.round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)

        return np.array(descriptors, dtype='float32')


############################ USAGE #############################
#sift = SIFT("ramsis.jpg")
#keypoints,descriptors = sift.computeKeypointsAndDescriptors()
