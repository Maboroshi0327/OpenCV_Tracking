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


def iou():
    pass
