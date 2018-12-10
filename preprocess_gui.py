import cv2
import numpy as np
from preprocess import PreProcess


def nothing(x):
    pass


class PreProcessBoard(PreProcess):

    winname = 'Preprocess parameter selection board'

    def __init__(self, pic_name, flags=0):
        PreProcess.__init__(self, pic_name, flags)

        cv2.namedWindow(winname=PreProcessBoard.winname, flags=0)
        cv2.resizeWindow(PreProcessBoard.winname, (500, 800))
        cv2.createTrackbar('kernel size', PreProcessBoard.winname, 0, 5, nothing)
        cv2.createTrackbar('bin thresh', PreProcessBoard.winname, 0, 255, nothing)

        while 1:
            s = cv2.getTrackbarPos('kernel size', PreProcessBoard.winname)
            thresh = cv2.getTrackbarPos('bin thresh', PreProcessBoard.winname)
            kernel = np.ones((s, s), np.uint8)
            img_mor = cv2.morphologyEx(self.img, op=cv2.MORPH_CLOSE, kernel=kernel)
            ret, img_bin = cv2.threshold(self.img, thresh, 255, cv2.THRESH_BINARY)
            # self.binarize(thresh, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow(PreProcessBoard.winname, img_mor)
            key = cv2.waitKey(1)
            if key == 27:
                break
            else:
                cv2.imshow(PreProcessBoard.winname, img_bin)


if __name__ == '__main__':
    ppb = PreProcessBoard('./data/t1.jpg')
