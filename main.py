import cv2

from ReadFiles import read_bounding_box, read_frames


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
        _, bbox = tracker.update(frame)
        print(index + 1)
        print(bboxes[index + 1])
        print(bbox)
        print()

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(name='SuperMario_ce', difficulty='hard')
