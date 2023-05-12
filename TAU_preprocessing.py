import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from TAU_img_functions import *

from interfaces import TAUPreprocessingInterface


class TAUPreprocessing(TAUPreprocessingInterface):
    '''
    Implements the image preprocessing, including image rescaling and cable starting points determination
    '''

    def __init__(self, input_img: np.ndarray, cable_D: float, n_cables: int, con_points: list, con_dim: float, cable_length: float, pixel_D: int = 4):
        self.img = input_img
        self.pixel_D = pixel_D
        self.cable_D = cable_D
        self.n_cables = n_cables
        self.con_points = con_points
        self.con_dim = con_dim
        self.cable_length = cable_length


    def exec(self) -> Tuple[np.ndarray, np.ndarray, list, list, float, list]:

        mm_per_pixel_original = self.con_dim/points_dist2D(self.con_points[0], self.con_points[1])
        scale = (self.cable_D/mm_per_pixel_original)/self.pixel_D

        cable_length_px = int(self.cable_length/mm_per_pixel_original)
        min_row = 0
        max_row = self.img.shape[0]
        min_col = max(min(self.con_points[0][1], self.con_points[1][1]) - 15, 0)
        max_col = min(min_col + cable_length_px, self.img.shape[1])
        crop_img = copy.deepcopy(self.img[min_row:max_row, min_col:max_col])

        n_row = int(crop_img.shape[0]/scale)
        n_col = int(crop_img.shape[1]/scale)
        resized_img = cv.resize(crop_img, (n_col, n_row), interpolation=cv.INTER_AREA)

        con_points_resized = []
        for point in self.con_points:
            con_points_resized += [[int((point[0]-min_row)/scale), int((point[1]-min_col)/scale)]]
        mm_per_pixel = self.con_dim/points_dist2D(con_points_resized[0], con_points_resized[1])

        init_points = []
        for i in range(1,self.n_cables+1,1):
            init_points += [[int(con_points_resized[0][0]-((self.con_dim/(self.n_cables+1))*i)/mm_per_pixel), int(con_points_resized[0][1]+((con_points_resized[1][1] - con_points_resized[0][1])/self.n_cables)*i)]]

        img_init_points = copy.deepcopy(resized_img)
        img_init_points = cv.rectangle(img_init_points, (con_points_resized[0][1] - 4, con_points_resized[0][0] - 4), (con_points_resized[0][1] + 4, con_points_resized[0][0] + 4), 255, 2)
        img_init_points = cv.rectangle(img_init_points, (con_points_resized[1][1] - 4, con_points_resized[1][0] - 4), (con_points_resized[1][1] + 4, con_points_resized[1][0] + 4), 255, 2)
        for init in init_points:
            img_init_points = cv.rectangle(img_init_points, (init[1] - 3, init[0] - 3), (init[1] + 3, init[0] + 3), [0,255,0], 2)

        windows_size = [int(self.pixel_D*1.5), int(self.pixel_D*2)]

        return resized_img, img_init_points, con_points_resized, init_points, mm_per_pixel, windows_size