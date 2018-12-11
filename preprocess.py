import cv2
import numpy as np


class PreProcess(object):
    '''preprocess pictures'''

    def __init__(self, pic_name, flags=0):
        self.pic_name = pic_name
        self.img = cv2.imread(self.pic_name, flags)
        self.copy = self.img.copy()

    def binarize(self, thresh, maxval, type):
        # ret_thresh1, self.img = cv2.threshold(self.img, thresh, maxval, type)
        ret_thresh2, self.img = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY)

    def show(self):
        cv2.imshow("Current Picture", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def morph_ex(self, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8)):
        # kernel = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        self.img = cv2.morphologyEx(self.img, op=cv2.MORPH_OPEN, kernel=kernel)

    def gauss_blur(self, size):
        self.img = cv2.blur(self.img, (size, size))

    def lambda_binary(self, lamda):
        self.img = pow(self.img/255, lamda)*255

    def restore(self):
        self.img = self.copy.copy()

    def morph(self, erosion=True, kernel=np.array([[1, 1], [1, 1]], dtype=np.uint8)):
        if erosion:
            self.img = cv2.erode(self.img, kernel=kernel)
        else:
            self.img = cv2.dilate(self.img, kernel=kernel)

8
if __name__ == '__main__':
    pp = PreProcess(pic_name='./data/t1.jpg', flags=0)
    pp.show()
    # pp.binarize(100, 255, cv2.THRESH_BINARY)
    # pp.binarize(100, 255, cv2.THRESH_BINARY)
    pp.lambda_binary(0.001)
    pp.show()



