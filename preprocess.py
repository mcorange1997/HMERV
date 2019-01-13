import cv2
import numpy as np
import sys
import os

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

    def median_filter(self, size=3):
        self.img = cv2.medianBlur(self.img, size)


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

    # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
    # laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    laplacian = cv2.Canny(gray, 100, 200)
    #    Sobel算子，y方向求梯度
    # sobel = cv2.Sobel(sobel, cv2.CV_8U, 0, 1, ksize=3)

    # 2. 二值化
    # s1 = 255 - gray
    # s1 = (pow(sobel/255, 0.3)*255).astype(np.uint8)
    # ret1, s1 = cv2.threshold(s1, 127, 255, cv2.THRESH_BINARY)
    # s1 = 255 - s1
    #
    # s2 = 255 - gray
    # s2 = (pow(sobel/255, 0.45)*255).astype(np.uint8)
    # ret2, s2 = cv2.threshold(s2, 127, 255, cv2.THRESH_BINARY)
    # s2 = 255 - s2
    #    binary = pow(sobel/255, 0.45)*255
    # binary = binary.astype(np.uint8)
    # s3 = (s1+s2) / 2
    # s3 = s3.astype(np.uint8)
    ret, binary = cv2.threshold(laplacian, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)


    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    element4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)
    erosion = cv2.morphologyEx(erosion, op=cv2.MORPH_OPEN, kernel=element4, iterations=1)
    # erosion = cv2.erode(erosion, element4, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)
    # dilation3 = cv2.dilate(dilation2, np.ones((2, 2), dtype=np.uint8), iterations=3)
    # cv2.imwrite('dilation3.png', dilation3)
    # 7. 存储中间图片
    cv2.imwrite("laplacian.png", laplacian)
    # cv2.imwrite("sobel.png", sobel)
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 400:
            continue

        # 轮廓近似，作用很小
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(approx)
        print("rect is: ")
        print(rect)

        # box是四个点的坐标
        angle = rect[1]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        # if area < 1000 and (0.6 > width/height or width/height > 1.5):
        #     continue

        region.append(box)

    return region


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    i = 0
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        x1, y1 = box.min(axis=0)
        x2, y2 = box.max(axis=0)

        filename = "pic%d.jpg"%i
        i += 1
        temp = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join('result', filename), temp)

        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    # 带轮廓的图片
    cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


if __name__ == '__main__':
    # 读取文件
    # imagePath = sys.argv[1]
    imagePath = 'data/t5.jpg'
    img = cv2.imread(imagePath)
    # detect(img)
    # img = cv2.imread('data/t4.jpg')

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
if __name__ == '__main__':
    pp = PreProcess(pic_name='./data/t1.jpg', flags=0)
    pp.show()
    # pp.binarize(100, 255, cv2.THRESH_BINARY)
    # pp.binarize(100, 255, cv2.THRESH_BINARY)
    pp.median_filter(5)
    pp.show()
'''


