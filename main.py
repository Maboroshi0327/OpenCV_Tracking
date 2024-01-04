import cv2
import numpy as np

from ReadFiles import read_bounding_box, read_frames
from ImageProcess import draw_bounding_box


def iou(bbox1, bbox2):
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)


def main(name='Boy', difficulty='easy'):
    # Read Bounding Box
    bboxes = read_bounding_box(name=name, difficulty=difficulty)
    first_box = bboxes[0]
    print(first_box)

    # Read Frames
    frames = read_frames(name=name, difficulty=difficulty)
    first_img = next(frames)

    # Initial tracker
    tracker = cv2.legacy.TrackerCSRT.create()
    tracker.init(first_img, first_box)

    # Update tracker
    for index, frame in enumerate(frames):
        print(index + 1)

        _, bbox = tracker.update(frame)

        draw_bounding_box(frame, bboxes[index + 1], bbox, (0, 0, 255),(255, 0, 0))

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(name='Torus', difficulty='easy')
