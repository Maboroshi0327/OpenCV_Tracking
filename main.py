import cv2

from ReadFiles import read_bounding_box, read_frames
from ImageProcess import draw_bounding_box, calculate_iou


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

        _, bbox_p = tracker.update(frame)
        bbox_y = bboxes[index + 1]

        draw_bounding_box(frame, bbox_y, bbox_p, (0, 0, 255), (255, 0, 0))
        iou = calculate_iou(bbox_y, bbox_p)
        print(f'iou: {iou}')

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(name='Cup', difficulty='easy')
