from decimal import Decimal
import os

import cv2
import numpy as np


def read_bounding_box(name='Boy', difficulty='easy'):
    path = f'./AVC_DATASET/{difficulty}/{name}/{name}_gt.txt'

    # Open [name]_gt.txt
    with open(path, 'r') as f:
        data = f.readlines()

        # '123,456,789,1234\n' -> 123,456,789,1234
        for i in range(len(data)):
            # '123,456,789,1234\n' -> '123','456','789','1234'
            data[i] = data[i][:-1].split(',')

            # '123' -> 123; '456' -> 456; ...
            for j in range(len(data[i])):
                data[i][j] = int(float(data[i][j]))

        return np.array(data)


def read_frames(name='Boy', difficulty='easy'):
    path = f'./AVC_DATASET/{difficulty}/{name}/img/'
    for file in os.listdir(path):
        yield cv2.imread(path + file)
