import time
import csv

import cv2
import numpy as np

from ReadFiles import read_bounding_box, read_frames
from ImageProcess import draw_bounding_box, calculate_iou


def tracking(name='Boy', difficulty='easy'):
    # Read Bounding Box
    bboxes_y = read_bounding_box(name=name, difficulty=difficulty)
    first_box = bboxes_y[0]

    # Read Frames
    frames = read_frames(name=name, difficulty=difficulty)
    first_img = next(frames)

    # Initial tracker
    tracker = cv2.legacy.TrackerCSRT.create()
    tracker.init(first_img, first_box)

    iou_arr = []

    start_time = time.time()

    # Update tracker
    for index, frame in enumerate(frames):
        _, bbox_p = tracker.update(frame)

        # Calculate IoU
        bbox_y = bboxes_y[index + 1]
        iou = calculate_iou(bbox_y, bbox_p)
        iou_arr.append(iou)

        # Show result
        # draw_bounding_box(frame, bbox_y, bbox_p, (0, 0, 255), (255, 0, 0))
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

    end_time = time.time()

    # Calculate iou_avg and avg_time
    iou_arr = np.array(iou_arr)
    iou_avg = np.mean(iou_arr) * 100
    avg_time = (end_time - start_time) / iou_arr.shape[0]

    return iou_avg, avg_time


def main():
    name = [['Boy', 'CarDark', 'Cup', 'Sunshade', 'Torus'],
            ['Carchasing_ce4', 'Fish_ce2', 'Hurdle_ce2', 'Iceskater', 'Tiger1'],
            ['Girlmov', 'Matrix', 'Panda', 'Skiing', 'SuperMario_ce']]
    difficulty = ['easy', 'medium', 'hard']

    # Output Average IoU and Average Time to csv file
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for i in range(3):
            title = [difficulty[i]]
            iou_avg = ['Average IoU']
            time_avg = ['Time (sec.)']

            # Tracking
            for j in name[i]:
                iou, t_avg = tracking(name=j, difficulty=difficulty[i])
                print(difficulty[i], j)
                print(f'iou_avg: {iou} %')
                print(f'avg_time: {t_avg} s')
                print()

                # Append data to list
                title.append(j)
                iou_avg.append(iou)
                time_avg.append(t_avg)

            # Write data to csv
            writer.writerow(title)
            writer.writerow(iou_avg)
            writer.writerow(time_avg)


if __name__ == "__main__":
    main()
