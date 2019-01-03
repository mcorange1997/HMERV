import cv2
import numpy as np
from skimage import morphology
import sys
import os


def showImg(winname, img):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def remove_iso_noise(img, radius=4):
    pass


def preprocess(img):
    '''
    预处理图像
    :param img:原图
    :return: 处理后的图片
    '''
    # 1. 变成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. 求出边缘
    canny = cv2.Canny(gray, 100, 200)
    s1 = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
    s2 = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
    canny = s1
    # canny = cv2.Sobel(canny, cv2.CV_8U, 0, 1, ksize=3)

    # 3. 二值化
    _, binary = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 形态学运算的kernel
    # k1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(20, 8))
    # k2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10, 3))
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(6, 6))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(4, 4))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(22, 10))
    k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(7, 7))

    # imglab = morphology.label(binary)
    # cleaned = morphology.remove_small_objects(imglab, min_size=4, connectivity=2)
    # img3 = np.zeros((cleaned.shape))  # create array of size cleaned
    # img3[cleaned > 0] = 255
    # img3 = np.uint8(img3)
    # showImg('img3', img3)

    # print(binary.shape)
    # for i in range(binary.shape[0]):
    #     for j in range(binary.shape[1]):
    #         if i >= 2 and j >= 2 and binary[i][j] == 255:
    #             area_check = binary[i-2:i+2, j-2:j+2]
    #             num = np.count_nonzero(area_check)
    #             print(num)
    #             if num < 6:
    #                 binary[i][j] = 0


    # 4. 先做一次膨胀，找到轮廓
    dilate1 = cv2.dilate(binary, kernel=k2, iterations=1)

    # 5. 一次开运算，断开连接
    # open1 = cv2.morphologyEx(dilate1, op=cv2.MORPH_OPEN, kernel=k2, iterations=2)
    # close1 = cv2.morphologyEx(open1, op=cv2.MORPH_CLOSE, kernel=k3, iterations=3)
    erosion1 = cv2.erode(dilate1, kernel=k4, iterations=1)

    # 6. 再一次膨胀，找到轮廓
    dilate2 = cv2.dilate(erosion1, kernel=k3, iterations=2)

    cv2.imwrite('segment/gray.jpg', gray)
    cv2.imwrite('segment/s1.jpg', s1)
    cv2.imwrite('segment/s2.jpg', s2)
    cv2.imwrite('segment/canny.jpg', canny)
    cv2.imwrite('segment/binary.jpg', binary)
    cv2.imwrite('segment/dilate1.jpg', dilate1)
    cv2.imwrite('segment/erosion1.jpg', erosion1)
    # cv2.imwrite('segment/close1.jpg', close1)
    cv2.imwrite('segment/dilate2.jpg', dilate2)

    return dilate2.astype(np.uint8)


def segment(pre, src):
    '''
    在原图上圈出分割效果
    :param pre: 预处理后的图片
    :param src: 原图片
    :return: 原图上圈出后的分割结果
    '''
    _, contours, _ = cv2.findContours(pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 200:
            continue

        rect = cv2.minAreaRect(contour)
        print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)
        cv2.drawContours(src, [box], 0,  (0, 0, 255), 3)

    return src


def shape(pre, src, canny):
    polygens = []
    _, contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        rect = cv2.minAreaRect(contour)
        bbox = cv2.boxPoints(rect)
        bbox = np.int0(bbox)
        epsilon = cv2.arcLength(bbox, closed=True)
        polygen = cv2.approxPolyDP(contour, epsilon=epsilon*0.01, closed=True)
        polygens.append(polygen)
        #cv2.drawContours(canny, [polygen], 0, color=(0, 0, 255), thickness=2)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canny, polygens, 0, color=(0, 0, 255), thickness=3)
    showImg('1', canny)


def statistics(pre):
    '''
    返回预处理后图片的每一行及每一列的灰度值和
    :param pre: 输入图片
    :return: 返回的统计信息
    '''
    stat_col = np.sum(pre, axis=0)
    stat_row = np.sum(pre, axis=1)
    print('stat_col', stat_col)
    return stat_col, stat_row


def find_blank(stat, num, thresh, non0_stat, avg_intensity):
    '''
    找出认为是划分点（空白）的地方
    寻找标准：连续num列或行的灰度值之和大于某一阈值thresh
    :param stat: 行或列的统计信息
    :param num: 连续的数量
    :param thresh: 阈值
    :return: 集合， 每个间隔的起始和结束坐标（索引）
    '''
    # reshape输入信息，方便之后访问
    stat = stat.reshape(1, -1)
    non0_stat = non0_stat.reshape(1, -1)

    # 输入统计信息长度
    n = len(stat[0])
    # 输入统计信息是否超过阈值的TF矩阵
    s = stat < thresh
    # print(s)
    print(non0_stat)
    print(s)
    c1, c2 = 0, 1
    finding = False
    seg_pos = set()
    for i in range(n):
        if finding and s[0][i] and stat[0][i]/non0_stat[0][i] <= avg_intensity:
            c2 = i
        elif (not finding) and s[0][i] and stat[0][i]/non0_stat[0][i] <= avg_intensity:
            finding = True
            c1 = i
        else:
            # not s[i]
            finding = False
            c2 = i-1

        if s[0][c1] and s[0][c2] and (not finding) and (c2-c1 >= num):
            seg_pos.add((c1, c2))
    print(seg_pos)
    return seg_pos


def SegRem_by_blank(img, pre, num, thresh, avg_intensity=75):
    statCol, statRow = statistics(pre)
    col_non0, row_non0 = np.nonzero(statCol), np.nonzero(statRow)
    c1, cn = col_non0[0][0], col_non0[0][-1]
    r1, rn = row_non0[0][0], row_non0[0][-1]
    h = img.shape[0]
    non0_stat = np.count_nonzero(pre, axis=0)
    seg_pos = find_blank(statCol, num, thresh, non0_stat, avg_intensity=avg_intensity)

    for p in seg_pos:
        cv2.line(img, ((p[0]+p[1])//2, 0), ((p[0]+p[1])//2, h), (0, 0, 255))

    cropImg, cropPre = img[r1:rn, c1:cn], pre[r1:rn, c1:cn]

    cv2.imshow('img', cropImg)
    cv2.imshow('pre', cropPre)
    cv2.imwrite('analysis/img.jpg', img)
    cv2.imwrite('analysis/pre.jpg', pre)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropImg, cropPre


def remove_thin_lines(pre):
    # lines = cv2.HoughLines(pre, 1, np.pi/360)

    lines = cv2.HoughLinesP(pre, 0.1, np.pi/3000, threshold=20, minLineLength=5, maxLineGap=100)
    # a, b, c = lines.shape
    # print('lineeeeeeeee', lines)
    if lines is not None:
       for pp in lines:
           if abs(pp[0][1] - pp[0][3]) < 60:
                cv2.line(pre, (pp[0][0], pp[0][1]), (pp[0][2], pp[0][3]), (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('remove line', pre)
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


def detect_angle(src, pre):
    lines = cv2.HoughLines(pre, 1, np.pi/360, 150)
    # for line in lines:
    #     print(line)
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 找出最多的倾斜角，作为图片的倾斜角，并以此旋转图片
    if lines is not None:
        theta, counts = np.unique(lines[:, 0, 1:], return_counts=True)  # 统计所有角度出现的次数
        rotate_theta = theta[np.argmax(counts)]
    else:
        rotate_theta = 0
    # cv2.imshow('before', src)
    # src = rotate_bound(src, 90-rotate_theta/np.pi*180)
    # pre = rotate_bound(pre, 90-rotate_theta/np.pi*180)
    # cv2.imshow('after', src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.imshow('1', src)
    #     cv2.line(pre, (x1, y1), (x2, y2), (0, 0, 0), 2)
    #     src = rotate_bound(src, 90-theta/np.pi*180)
    #     cv2.imshow('2', src)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # cv2.imwrite('segment/lines.jpg', pre)

    return rotate_theta


if __name__ == '__main__':
    src = cv2.imread('data/t3.jpg')
    # pre = preprocess(src)

    # pre = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # pre = cv2.Canny(src, 120, 220)
    pre = preprocess(src)
    showImg('preprocess', pre)
    # cv2.imwrite('canny.jpg', pre)
    # detect_angle(src, pre)
    seg = segment(pre, src)
    cv2.imwrite('segment/contour.jpg', seg)
    # remove_thin_lines(pre)
    angle = detect_angle(src, pre)
    pre = rotate_bound(pre, angle)
    src = rotate_bound(src, angle)
    # remove_thin_lines(pre)

    # SegRem_by_blank(src, pre, 2, 800, 75)


    # cv2.imshow('11111', pre)
    # SegRem_by_blank(src, pre, 3, 1000)






