import cv2
import numpy as np
import sys
import os


def preprocess(img):
    # 1. 变成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. 求出边缘
    canny = cv2.Canny(gray, 100, 200)

    # 3. 二值化
    _, binary = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 形态学运算的kernel
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2, 2))

    # 4. 先做一次膨胀，找到轮廓
    dilate1 = cv2.dilate(binary, kernel=k1, iterations=1)

    # 5. 一次开运算，断开连接
    open1 = cv2.morphologyEx(dilate1, op=cv2.MORPH_OPEN, kernel=k2, iterations=3)
    # close1 = cv2.morphologyEx(open1, op=cv2.MORPH_CLOSE, kernel=k3, iterations=3)

    # 6. 再一次膨胀，找到轮廓
    dilate2 = cv2.dilate(open1, kernel=k3, iterations=1)

    cv2.imwrite('segment/canny.jpg', canny)
    cv2.imwrite('segment/binary.jpg', binary)
    cv2.imwrite('segment/dilate1.jpg', dilate1)
    cv2.imwrite('segment/open1.jpg', open1)
    # cv2.imwrite('segment/close1.jpg', close1)
    cv2.imwrite('segment/dilate2.jpg', dilate2)

    return dilate2.astype(np.uint8)


def segment(pre, src):
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


def stat_col_row(pre):
    stat_col = np.sum(pre, axis=0)
    stat_row = np.sum(pre, axis=1)
    return stat_col, stat_row


def find_blank(stat, num, thresh):
    '''
    找出空格
    stat是列或行统计信息
    num是判断连续多少小于阈值
    '''
    stat = stat.reshape(1, -1)
    print('shape', stat.shape)
    n = len(stat[0])
    s = stat < thresh
    print(s)

    c1, c2 = 0, 0
    finding = False
    seg_pos = set()
    for i in range(n):
        if finding and s[0][i]:
            c2 = i
        elif (not finding) and s[0][i]:
            finding = True
            c1 = i
        else:
            # not s[i]
            finding = False

        if s[0][c1] and s[0][c2] and (not finding) and (c2-c1>=3):
            seg_pos.add((c1, c2))

    return seg_pos


if __name__ == '__main__':
    src = cv2.imread('result/pic1.jpg')
    pre = preprocess(src)
    seg = segment(pre, src)
    cv2.imwrite('segment/contour.jpg', seg)

    stat_col, stat_row = stat_col_row(pre)
    seg_pos = find_blank(stat_col, 6, 12000)
    print('------------\n', seg_pos)

    h = src.shape[0]

    for point in seg_pos:
        cv2.line(src, ((point[0]+point[1])//2, 0), ((point[0]+point[1])//2, h), (0, 0, 255))

    cv2.imshow('line', src)


    col_non0 = np.nonzero(stat_col)
    row_non0 = np.nonzero(stat_row)
    c1, cn = col_non0[0][1], col_non0[0][-1]
    r1, rn = row_non0[0][1], row_non0[0][-1]

    final = src[r1:rn, c1:cn]
    cv2.imwrite('segment/final.jpg', final)

    cv2.imshow('img', seg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




