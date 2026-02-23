import cv2
import numpy as np
import os

def extract_predicted_boundary_mask(result_image):

    # Red pixels detection (BGR)
    red = result_image[:,:,2]
    green = result_image[:,:,1]
    blue = result_image[:,:,0]

    mask = np.zeros(red.shape, dtype=np.uint8)

    mask[(red > 150) & (green < 100) & (blue < 100)] = 1

    return mask


def compute_boundary_score(pred_image_path, gt_mask_path):

    pred_img = cv2.imread(pred_image_path)
    gt_mask = cv2.imread(gt_mask_path, 0)

    gt_mask = (gt_mask > 127).astype(np.uint8)

    pred_mask = extract_predicted_boundary_mask(pred_img)

    intersection = np.sum(pred_mask * gt_mask)
    total_pred = np.sum(pred_mask)

    if total_pred == 0:
        return 0.0

    score = intersection / total_pred

    return score