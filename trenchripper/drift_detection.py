import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from sklearn import linear_model

def find_seed_image(image_stack, max_poi_std=5):
  # Find a candidate image in the stack (t x y x x)
  num_points = image_stack.shape[0]
  for i in range(num_points-1):
    _, _, std_distance = get_orb_pois(image_stack[i], image_stack[i+1])
    if std_distance < max_poi_std:
      seed_index = i
      break
    if i == num_points-2:
      raise Exception("Could not find a suitable seed image in the timeseries")
  return seed_index

def get_orb_pois(im1, im2, max_interest_points= 500, match_threshold=0.1):
    orb = cv2.ORB_create(max_interest_points)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
    matches = matcher.match(descriptors1, descriptors2, None)
  # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

  # Remove not so good matches
    numGoodMatches = int(len(matches) * match_threshold)
    matches = matches[:numGoodMatches]
    distances = [match.distance for match in matches]
    std_distance = np.std(distances)
    
  # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    return points1, points2, std_distance

def find_drift_poi(points1, points2):
  # Find homography
    h, _ = cv2.estimateAffinePartial2D(points1,points2)
    return np.array([h[0,2], h[1,2]])
  
def find_template(reference, check, match_threshold=0.1,window_height=400, window_width=200):
    points1, points2, _ = get_orb_pois(reference, check, match_threshold=match_threshold)
    median_position = np.median(points1, axis=0)
    top_left = np.array([int(median_position[0]-window_width/2), int(median_position[1]-window_height/2)])
    template = reference[top_left[1]:top_left[1]+window_height,top_left[0]:top_left[0]+window_width]
    return template, top_left

def find_drift_template(template, top_left, im2):
    res = cv2.matchTemplate(im2, template, cv2.TM_SQDIFF_NORMED)
    _, _, top_left_new, _ = cv2.minMaxLoc(res)
    return top_left_new - top_left