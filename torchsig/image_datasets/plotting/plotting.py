import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_yolo_boxes_on_image(image, labels):
    image = 1 - image.numpy()
    image = (np.stack([image[0,:,:]]*3).transpose(1,2,0)*255).astype(np.uint8)
    for label in labels:
        cid, cx, cy, w, h = label
        img_h, img_w = image.shape[:2]
        x1 = int((cx - w/2)*img_w)
        x2 = int((cx + w/2)*img_w)
        y1 = int((cy - h/2)*img_h)
        y2 = int((cy + h/2)*img_h)
        image = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color=(255,0,0), thickness=1)
    plt.imshow(image)

def plot_yolo_datum(yolo_datum):
    plot_yolo_boxes_on_image(yolo_datum.img, yolo_datum.labels)