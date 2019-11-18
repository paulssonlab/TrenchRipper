import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from sklearn import linear_model
from skimage.util import pad

def find_seed_image(image_stack, max_poi_std=5):
    """
    
    """
    num_points = image_stack.shape[0]
    for i in range(num_points-1):
        _, _, std_distance = get_orb_pois(image_stack[i], image_stack[i+1])
        if std_distance < max_poi_std:
            seed_index = i
            break
        if i == num_points-2:
            raise Exception("Could not find a suitable seed image in the timeseries")
    return seed_index

def scale_image(im):
    min_intensity = np.min(im, axis=None)
    max_intensity = np.max(im, axis=None)
    scaling_factor = 255/(max_intensity - min_intensity)
    im_rescaled = (im-min_intensity)*scaling_factor
    im_rescaled = im_rescaled.astype(np.uint8)
    return im_rescaled

def get_orb_pois(im1, im2, max_interest_points= 750, match_threshold=0.1, plot=False, pad_width=15):
    orb = cv2.ORB_create(max_interest_points, patchSize=31)
    im1_scaled = scale_image(im1)
    im2_scaled = scale_image(im2)
    im1_scaled = pad(im1_scaled, pad_width=pad_width, mode='reflect')
    im2_scaled = pad(im2_scaled, pad_width=pad_width, mode='reflect')
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
    std_distance = np.std(distances)
    
    if plot:
        im3 = cv2.drawMatches(im1_scaled, keypoints1, im2_scaled, keypoints2, matches, None)
        plt.figure(figsize=(20, 10))
        plt.imshow(im3)
        plt.show()

    
  # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    points1 -= pad_width
    points2 -= pad_width
    return points1, points2, std_distance

def find_drift_poi(points1, points2):
  # Find homography
    h, _ = cv2.estimateAffinePartial2D(points1,points2)
    return np.array([h[0,2], h[1,2]])
  
def find_template(reference, check, match_threshold=0.1,window_height=400, window_width=200):
    reference_scaled = scale_image(reference)
    check_scaled = scale_image(check)
    points1, points2, _ = get_orb_pois(reference_scaled, check_scaled, match_threshold=match_threshold)
    median_position = np.median(points1, axis=0)
    top_left = np.array([max(0, int(median_position[0]-window_width/2)), max(0, int(median_position[1]-window_height/2))])
    template = reference_scaled[top_left[1]:top_left[1]+window_height,top_left[0]:top_left[0]+window_width]
    return template, top_left

def find_drift_template(template, top_left, im2):
    im2_scaled = scale_image(im2)
    res = cv2.matchTemplate(im2_scaled, template, cv2.TM_SQDIFF_NORMED)
    min_sqdiff, _, top_left_new, _ = cv2.minMaxLoc(res)
    return top_left_new - top_left, min_sqdiff