import cv2
import numpy as np
from preprocess import PreProcess


def nothing(x):
    pass


class PreProcessBoard(PreProcess):

    winname = 'Preprocess parameter selection board'

    def __init__(self, pic_name, flags=0):
        PreProcess.__init__(self, pic_name, flags)

        cv2.namedWindow(winname=PreProcessBoard.winname, flags=1)
        h, w = self.img.shape
        print(self.img.shape)
        # cv2.resizeWindow(PreProcessBoard.winname, int(w/2), int(h/2))
        cv2.createTrackbar('kernel size', PreProcessBoard.winname, 0, 5, nothing)
        cv2.createTrackbar('bin thresh', PreProcessBoard.winname, 127, 255, nothing)
        cv2.createTrackbar('lambda', PreProcessBoard.winname, 0, 10000, nothing)

        while 1:
            s = cv2.getTrackbarPos('kernel size', PreProcessBoard.winname)
            thresh = cv2.getTrackbarPos('bin thresh', PreProcessBoard.winname)
            kernel = np.ones((s, s), np.uint8)
            lamda = cv2.getTrackbarPos('lambda', PreProcessBoard.winname)

            # img_mor = cv2.morphologyEx(self.img, op=cv2.MORPH_CLOSE, kernel=kernel)
            # ret, img_bin = cv2.threshold(self.img, thresh, 255, cv2.THRESH_BINARY)
            self.restore()
            self.lambda_binary(lamda/100.0)
            self.binarize(thresh, 255, type=cv2.THRESH_BINARY)
            self.img = 255 - self.img
            # self.gauss_blur(2)
            self.morph(True)
            self.img = 255 - self.img
            # self.binarize(thresh, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow(PreProcessBoard.winname, img_mor)
            key = cv2.waitKey(1)
            if key == 27:
                break
            else:
                cv2.imshow(PreProcessBoard.winname, self.img)


if __name__ == '__main__':
    ppb = PreProcessBoard('./data/t1.jpg')
