import cv2
import numpy as np
import os

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def apply_morphology(img):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

def update_threshold(image, value):
    ret, thresh = cv2.threshold(sharpen_image(image), value, 255, cv2.THRESH_BINARY_INV)
    closed_thresh = apply_morphology(thresh)
    contours, hierarchy = cv2.findContours(closed_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
    cv2.putText(image_contours, f'Total Contours: {len(contours)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Contours', cv2.resize(image_contours, (0, 0), fx=0.5, fy=0.5))
    count_clusters(closed_thresh, image_contours)

def count_clusters(thresh, image_contours, circularity_threshold=0.7):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cluster_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > circularity_threshold:
            cluster_count += 1
            cv2.drawContours(image_contours, [contour], -1, (0, 255, 0), 2)
    cv2.putText(image_contours, f'Clusters: {cluster_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Clusters', cv2.resize(image_contours, (0, 0), fx=0.5, fy=0.5))

def process_images_in_directory(directory_path, threshold_value):
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                print(f'Processing {filename}')
                update_threshold(image, threshold_value)
            else:
                print(f'Failed to load {filename}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

directory_path = '/Users/birajlayek1230/Desktop/task/CountingChallenge/nuts'
threshold_value = 105

process_images_in_directory(directory_path, threshold_value)
