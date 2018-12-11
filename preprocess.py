import cv2
import numpy as np


class PreProcess(object):
    '''preprocess pictures'''

    def __init__(self, pic_name, flags=0):
        self.pic_name = pic_name
        self.img = cv2.imread(self.pic_name, flags)
        self.copy = self.img.copy()

    def binarize(self, thresh, maxval, type):
        #  ret_thresh1, self.img = cv2.threshold(self.img, thresh, maxval, type)
        ret_thresh2, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    def show(self):
        cv2.imshow("Current Picture", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def morphEX(self, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8)):
        self.img = cv2.morphologyEx(self.img, op=cv2.MORPH_OPEN, kernel=kernel)

    def gauss_blur(self, size):
        self.img = cv2.blur(self.img, (size, size))


if __name__ == '__main__':
    pp = PreProcess(pic_name='./data/t1.jpg', flags=0)
    pp.show()
    pp.binarize(100, 255, cv2.THRESH_BINARY)
    pp.show()
    pp.img.dilated
    pp.show()



