import cv2 as cv
from TAU_segmentation import TAUSegmentation
from TAU_preprocessing import TAUPreprocessing
from TAU_forward_propagation import TAUForwardPropagation
from TAU_backward_propagation import TAUBackwardPropagation
from TAU_cable_line_estimation2 import TAUCableLineEstimation
from TAU_critique import TAUCritique
import copy
import numpy as np
import os

show_imgs = True

class DLO_estimator():

    def __init__(self, img, all_colors, color_order, con_points, cable_D, con_dim, cable_lengths, pixel_D):

        self.red_color_paint = [0,0,255]
        self.background_color = [221,160,221]
        self.color_order = color_order
        self.pixel_D = pixel_D
        self.cable_lengths = cable_lengths

        self.preprocessing = TAUPreprocessing(input_img = img, cable_D = cable_D, n_cables = n_cables, con_points = con_points, con_dim=con_dim, cable_length=max(cable_lengths), pixel_D=pixel_D)
        resized_img, img_init_points, con_points_resized, self.init_points, self.mm_per_pixel, self.window_size = self.preprocessing.exec()
        if show_imgs:
            cv.imshow("Initial points", img_init_points)
        init_col = min(con_points_resized[0][1], con_points_resized[1][1])
        
        self.segmentation = TAUSegmentation(input_img=resized_img, all_colors=all_colors, init_col = init_col)
        self.FWP = TAUForwardPropagation(self.mm_per_pixel)
        self.BWP = TAUBackwardPropagation(self.mm_per_pixel)
        self.line_estimation = TAUCableLineEstimation(max_order=8, min_order=0)

        #Creates a background image to draw the estimated cables
        bkg = np.zeros((resized_img.shape[0], resized_img.shape[1], 3), dtype='uint8')
        for row in range(bkg.shape[0]):
            for col in range(bkg.shape[1]):
                bkg[row][col] = self.background_color
        self.points_cables_img = copy.deepcopy(resized_img)
        self.lines_cables_img = copy.deepcopy(resized_img)
        self.lines_cables_bkg = copy.deepcopy(bkg)


    def exec(self, forward, iteration, thr1_init=60, thr2_init=1.5, erosion_init=3):

        index = 0
        all_points_cables = []
        for color in self.color_order:
            thr1 = thr1_init
            thr2 = thr2_init
            erosion = erosion_init
            WinSize = self.window_size
            success_cable = False

            i=0
            init_points_BW = [self.init_points[index]]
            for c in self.color_order:
                if i != index and c == color:
                    init_points_BW.append(self.init_points[i])
                i+=1
            n_cables = len(init_points_BW)
            cable_length = self.cable_lengths[index]

            if iteration:
                self.critique = TAUCritique(self.pixel_D, thr1, thr2, erosion, WinSize, forward)                

            ite=1
            while not success_cable:
                print("Cable " + str(index) + ": calculating (iteration " + str(ite) + ")...")
                ite+=1

                #Segmentation
                segm_img, segm_pixels  = self.segmentation.exec(color_cable = color, thr1=thr1, thr2=thr2, erosion=3)
                if show_imgs:
                    cv.imshow("Segmentation", segm_img)

                #Cable points calculation
                #FWP
                if forward:
                    points_cable, n_captured_points, success_points, count_no_borders, count_free_steps, init_success = self.FWP.exec(segm_img, segm_pixels, self.init_points[index], WinSize, cable_length)
                
                #BWP
                else:
                    points_cable, n_captured_points, success_points = self.BWP.exec(segm_img, segm_pixels, init_points_BW, n_cables, WinSize, cable_length)
                    init_success = True; count_no_borders = 0; count_free_steps = 0

                #Critique
                if iteration:
                    success_cable, result_error, thr1, thr2, erosion, WinSize = self.critique.exec(points_cable, segm_img, len(segm_pixels), n_captured_points, success_points, n_cables, thr1, thr2, WinSize, erosion, init_success, count_no_borders, count_free_steps)
                else:
                    success_cable = True

            #Cable polynomial function estimation
            cable_line_yx, cable_line_xy = self.line_estimation.exec(points_cable)
            all_points_cables.append(cable_line_yx)

            #Paint results
            for point in points_cable:
                self.points_cables_img = cv.rectangle(self.points_cables_img, (point[1]-self.window_size[1],point[0]-self.window_size[0]), (point[1]+self.window_size[1],point[0]+self.window_size[0]), [0,255,0], 2)
            for point1, point2 in zip(cable_line_xy, cable_line_xy[1:]): 
                self.lines_cables_img = cv.line(self.lines_cables_img, point1, point2, self.red_color_paint, 2)
                self.lines_cables_bkg = cv.line(self.lines_cables_bkg, point1, point2, color, 2)

            print("Cable " + str(index) + " CALCULATED")
            print("--------------------------------------------------")
            index += 1

        if show_imgs:
            cv.imshow("Points cables", self.points_cables_img)
                
        cv.imshow("Lines cables", self.lines_cables_img)
        cv.imshow("Lines cables bkg", self.lines_cables_bkg)

        return all_points_cables



if __name__ == "__main__":

    #img_path = os.path.dirname(os.path.realpath(__file__)) + '\\test_images\\wh1_multicolor.jpeg' 
    img_path = os.path.dirname(os.path.realpath(__file__)) + '/test_images/wh1_pinkblue.jpeg'
    print(img_path)
    img = cv.imread(img_path)
    if show_imgs:
        cv.imshow("Original image", img)

    yellow_cable = [40,141,171]
    blue_cable = [157,99,48]
    green_cable = [43,108,50]
    red_cable = [53,49,181]
    black_cable = [57,54,71]
    white_cable = [181,184,191]

    cable_colors = [yellow_cable, blue_cable, green_cable, red_cable, black_cable, white_cable]
    #order: from bottom to top
    cables_color_order = [black_cable, red_cable, white_cable, blue_cable, green_cable, yellow_cable, white_cable, blue_cable, green_cable, yellow_cable]
    cable_lengths = [550, 550, 550, 550, 550, 550, 550, 550, 550, 550]
    n_cables = len(cables_color_order)

    #All measures in mm
    #con_corner_below = [698, 107] #below corner for wh1_multicolor.jpeg image
    #con_corner_above = [472, 129] #above corner for wh1_multicolor.jpeg image
    con_corner_below = [500, 185] #below corner for wh1_pinkblue.jpeg image
    con_corner_above = [270, 185] #above corner for wh1_pinkblue.jpeg image
    con_dim = 27 #Connector width
    cable_D = 1.32 #cable diameter
    pixel_D = 4 #desired number of pixel per cable diameter

    estimate_dlos = True #If False just the connector corner pixels are visualized
    forward = False #forward or backward points propagation
    iteration = False #with or without feedback

    p = DLO_estimator(img=img, all_colors=cable_colors, color_order = cables_color_order, con_points=[con_corner_below, con_corner_above], cable_D=cable_D, con_dim=con_dim, cable_lengths=cable_lengths, pixel_D=pixel_D)
    if estimate_dlos:
        all_points_cables = p.exec(forward, iteration)

    cv.waitKey(0)