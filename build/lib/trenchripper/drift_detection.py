import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import pad

def find_seed_image(image_stack, max_poi_std=5):
    """ Find a suitable image for segmentation from an image stack by checking interest point
    matching to neighbouring timepoints (out of focus or very drifted timepoints should not
    get good matches). We measure this by looking at the standard deviation of interest point
    matches.

    Args:
        image_stack (numpy.ndarray, t x y x x): Image stack
        max_poi_std: Standard deviation in Euclidean distance between interest points
    Returns:
        seed_index (int): Timepoint to use as seed
    
    """
    num_points = image_stack.shape[0]
    for i in range(num_points-1):
        # Get standard deviation of interest point matches
        _, _, std_distance = get_orb_pois(image_stack[i], image_stack[i+1])
        # If it's suitable use this as a seed
        if std_distance < max_poi_std:
            seed_index = i
            break
        if i == num_points-2:
            raise Exception("Could not find a suitable seed image in the timeseries")
    return seed_index

def scale_image(im):
    """ Stretch image intensities to fill whole histogram and convert to 8bit for opencv

    Args:
        im: image to scale
    Returns:
        im_rescaled: scaled image

    """

    # Get min and max
    min_intensity = np.min(im, axis=None)
    max_intensity = np.max(im, axis=None)
    # Stretch histogram
    scaling_factor = 255/(max_intensity - min_intensity)
    im_rescaled = (im-min_intensity)*scaling_factor
    # Convert to 8bit
    im_rescaled = im_rescaled.astype(np.uint8)
    return im_rescaled

def get_orb_pois(im1, im2, max_interest_points= 750, match_threshold=0.1, plot=False, pad_width=15):
    """ Find matching points of interest between two images according to ORB algorithm

    Args:
        im1 (numpy.ndarray, 8-bit integer): first image (y x x)
        im2 (numpy.ndarray, 8-bit integer): second image (y x x)
        max_interest_points (int): Maximum number of points to recognize using ORB algorithm
        match_threshold (float): The fraction of closest matches in Euclidean distance 
    Returns:
        points1 (numpy.ndarray, int): List of filtered points in image 1 that matched image 2 (n x 2)
        points2 (numpy.ndarray, int): List of filtered points in image 2 that matched image 1 (n x 2)
        std_distance (float): Standard deviation of distances between the matched points
    """
    orb = cv2.ORB_create(max_interest_points, patchSize=31)
    # Scale images
    im1_scaled = scale_image(im1)
    im2_scaled = scale_image(im2)
    # Pad images so that interest points on edges can be recognized
    im1_scaled = pad(im1_scaled, pad_width=pad_width, mode='reflect')
    im2_scaled = pad(im2_scaled, pad_width=pad_width, mode='reflect')
    # Match points based on their ORB descriptors
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_scaled, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_scaled, None)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * match_threshold)
    matches = matches[:numGoodMatches]
    distances = [match.distance for match in matches]
    # Get std
    std_distance = np.std(distances)
    
    if plot:
        im3 = cv2.drawMatches(im1_scaled, keypoints1, im2_scaled, keypoints2, matches, None)
        plt.figure(figsize=(20, 10))
        plt.imshow(im3)
        plt.show()

    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # Get points 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Offset by padding
    points1 -= pad_width
    points2 -= pad_width
    return points1, points2, std_distance

def find_drift_poi(points1, points2):
    """ Find the best fit for the rotation and translation that map points1 to points2 using 
    the RANSAC algorithm (fitting robust to outliers).

    Args:
        points1 (numpy.ndarray, int): list of points (n x 2)
        points2 (numpy.ndarray, int): list of points (n x 2)
    Returns:
        array: x and y translation
    """
    # Find affine transformation matrix from points1 to points2 (i.e. Ax1 = x2)
    h, _ = cv2.estimateAffinePartial2D(points1,points2)
    # Extract x and y translation
    return np.array([h[0,2], h[1,2]])
  
def find_template(reference, check, match_threshold=0.1,window_height=400, window_width=200):
    """ Find template region (fiduciary markers on the mother machine) by cluster of interest points

    Args:
        reference (numpy.ndarray): Reference image from which to extract the template 
        check (numpy.ndarray): Check image to see that the template matches to another timepoint
        match_threshold (float): The fraction of closest matches in Euclidean distance 
        window_height (int): Height of template patch to extract
        window_width (int): Width of template patch to extract
    Returns:
        template (numpy.ndarray): Template match from reference image (window_height x window_width)
        top_left (numpy.ndarray): x and y coordinates of the top-left corner of the template (used to calculate drift) (1 x 2)
    """
    reference_scaled = scale_image(reference)
    check_scaled = scale_image(check)
    # Get interest points
    points1, points2, _ = get_orb_pois(reference_scaled, check_scaled, match_threshold=match_threshold)
    # Get median of interest points in the first image
    median_position = np.median(points1, axis=0)
    # Calculate top left
    top_left = np.array([max(0, int(median_position[0]-window_width/2)), max(0, int(median_position[1]-window_height/2))])
    # Crop image to get template
    template = reference_scaled[top_left[1]:top_left[1]+window_height,top_left[0]:top_left[0]+window_width]
    return template, top_left

def find_drift_template(template, top_left, im2):
    """ Calculate drift using template matching

    Args:
        template(numpy.ndarray): template patch
        top_left (numpy.ndarray): coordinates of top_left of template patch in reference image (1 x 2)
        im2 (numpy.ndarray): image to template match
    Returns:
        array (numpy.ndarray) : drift (1 x 2)
        min_sqdiff: Normalized least square difference between the template match
    """
    im2_scaled = scale_image(im2)
    res = cv2.matchTemplate(im2_scaled, template, cv2.TM_SQDIFF_NORMED)
    min_sqdiff, _, top_left_new, _ = cv2.minMaxLoc(res)
    return top_left_new - top_left, min_sqdiff