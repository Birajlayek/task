import cv2
import numpy as np
import os

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def apply_morphology(img):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

def update_threshold(value, image, image_contours):
    sharpened_image = sharpen_image(image)
    ret, thresh = cv2.threshold(sharpened_image, value, 255, cv2.THRESH_BINARY_INV)
    closed_thresh = apply_morphology(thresh)
    contours, hierarchy = cv2.findContours(closed_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
    cv2.putText(image_contours, f'Total Contours: {len(contours)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cluster_count = count_clusters(closed_thresh, image_contours)
    return image_contours, cluster_count

def count_clusters(thresh, image_contours):
    num_labels, labels_im = cv2.connectedComponents(thresh)
    cluster_count = 0
    min_area = 100
    for label in range(1, num_labels):
        mask = labels_im == label
        area = np.sum(mask)
        if area >= min_area:
            cluster_count += 1
    cv2.putText(image_contours, f'Clusters: {cluster_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cluster_count

folder_path = '/Users/birajlayek1230/Desktop/task/CountingChallenge/Bolts'  # Update with your folder path
threshold_value = 105

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        processed_image, cluster_count = update_threshold(threshold_value, image, image_contours)
        print(f'{filename}: {cluster_count} objects')
        cv2.imshow(f'Contours - {filename}', cv2.resize(processed_image, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
