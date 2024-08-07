import cv2
import numpy as np
import os

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_objects(img, net, output_layers):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

def draw_labels(img, class_ids, confidences, boxes, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    cluster_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if label == 'nut' else (0, 0, 255)
            if label == 'nut':
                cluster_count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    cv2.putText(img, f'Total Clusters: {cluster_count}', (10, 30), font, 2, (0, 0, 255), 2)
    return img

def process_images_in_directory(directory_path):
    net, output_layers, classes = load_yolo()
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                print(f'Processing {filename}')
                class_ids, confidences, boxes = detect_objects(image, net, output_layers)
                image = draw_labels(image, class_ids, confidences, boxes, classes)
                cv2.imshow('Image', cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(0)
            else:
                print(f'Failed to load {filename}')
    cv2.destroyAllWindows()

directory_path = '/Users/birajlayek1230/Desktop/task/CountingChallenge/nuts'
process_images_in_directory(directory_path)
