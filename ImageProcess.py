import cv2
import numpy as np


def draw_bounding_box(frame, bbox_y, bbox_p, color_y, color_p):
    p1 = (int(bbox_y[0]), int(bbox_y[1]))
    p2 = (int(bbox_y[0] + bbox_y[2]), int(bbox_y[1] + bbox_y[3]))
    cv2.rectangle(frame, p1, p2, color_y, 2, 1)

    p1 = (int(bbox_p[0]), int(bbox_p[1]))
    p2 = (int(bbox_p[0] + bbox_p[2]), int(bbox_p[1] + bbox_p[3]))
    cv2.rectangle(frame, p1, p2, color_p, 2, 1)

    return frame


def calculate_iou(box1, box2):
    # box format: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Calculate the area of intersection rectangle
    area_intersection = max(0, w_intersection) * max(0, h_intersection)

    # Calculate the area of both bounding boxes
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Calculate the Union (sum of both areas - intersection)
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate IoU
    iou = area_intersection / area_union

    return iou
