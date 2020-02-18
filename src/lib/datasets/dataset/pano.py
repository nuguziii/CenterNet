from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import glob


class PANODataset:
    def __init__(self, path):
        super(PANODataset, self).__init__()
        self.data_dir = path
        self.w_pad = 100
        self.h_pad = 0
        self.ratio = 2

    def getImgIds(self):
        file_list = glob.glob(self.data_dir + '/*.jpg')
        return file_list

    def _loadImg(self, file):
        return cv2.imread(file, cv2.IMREAD_COLOR)

    def loadData(self, file):
        original_img = self._loadImg(file)
        img, anns = self._cropImg(original_img, self._loadAnn(file))
        img, anns = self._resize(img, anns)
        img = self._normalize(img)

        #self.showImage(img, anns)
        return img, anns

    def getOriginalCoord(self, original, ann):
        width = np.size(original, 1)
        height = np.size(original, 0)

        if height * self.ratio < width:
            newH = height
            newW = height * self.ratio
        elif height * self.ratio > width:
            newH = int(width / self.ratio)
            newW = newH * self.ratio
        else:
            newH = height
            newW = width

        diffH = int((height - newH) / 2)
        diffW = int((width - newW) / 2)

        original = original[diffH:diffH + newH, diffW:diffW + newW, :]

        width = np.size(original, 1)
        height = np.size(original, 0)

        fx = width / 512
        fy = height / 256

        ann = (ann * np.array([fx, fy, fx, fy])).astype(int) + np.array([diffW - self.w_pad, diffH + self.h_pad, diffW - self.w_pad, diffH + self.h_pad])

        return ann

    def _normalize(self, img):
        inp = (img.astype(np.float32) / 255.)

        val = np.reshape(inp, (-1,3))
        mean = np.mean(val, axis=0).reshape(1, 1, 3)
        std = np.std(val, axis=0).reshape(1, 1, 3)

        inp = (inp - mean) / std
        return inp

    def _resize(self, img, anns):
        img_new = cv2.resize(img, (512, 256), interpolation=cv2.INTER_CUBIC)
        fx = np.size(img, 1) / 512
        fy = np.size(img, 0) / 256

        for ann in anns:
            ann["bbox"] = (ann["bbox"]/np.array([fx, fy, fx, fy])).astype(int)
            ann["center"] = (ann["center"]/np.array([fx, fy])).astype(int)

        return img_new, anns

    def _cropImg(self, img, anns):
        width = np.size(img, 1)
        height = np.size(img, 0)

        img = img[self.h_pad:height-self.h_pad, self.w_pad:width-self.w_pad, :]

        width = np.size(img, 1)
        height = np.size(img, 0)

        if height*self.ratio<width:
            newH = height
            newW = height*self.ratio
        elif height*self.ratio>width:
            newH = int(width/self.ratio)
            newW = newH*self.ratio
        else:
            newH = height
            newW = width

        diffH = int((height-newH)/2)
        diffW = int((width-newW)/2)
        img = img[diffH:diffH+newH,diffW:diffW+newW,:]

        for ann in anns:
            ann["bbox"] = ann["bbox"] - np.array([diffW + self.w_pad, diffH + self.h_pad, diffW + self.w_pad, diffH + self.h_pad])
            ann["center"] = ann["center"] - np.array([diffW + self.w_pad, diffH + self.h_pad])

        return img, anns

    def _loadAnn(self, file):
        '''
        :param file: pano jpg image path
        :return data: [ {"num":1, "bbox":[(0,0),(0,0),(0,0),(0,0)], "center":(0,0)} , {} , ... ]
        '''
        data = []

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        centerX = int(np.size(img, 1) / 2)
        centerY = int(np.size(img, 0) / 2)

        f = open(file.replace('jpg', 'txt'), 'r')
        txt = f.readlines()
        f.close()
        for t in txt:
            dict = {}
            t = t.replace('\n', '').split(', ')

            x_coord = []
            y_coord = []

            for i in range(8):
                x_coord.append(centerX + int(t[2 * i + 3]))
                y_coord.append(centerY + int(t[2 * i + 4]))

            coords = [min(x_coord), min(y_coord), max(x_coord), max(y_coord)]  # minX, minY, maxX, maxY

            dict["bbox"] = np.array([coords[0], coords[1], coords[2], coords[3]])  # from top-left corner, clockwise
            dict["num"] = t[0]
            dict["center"] = np.array([centerX + int(t[27]), centerY + int(t[28])])
            if t[1]=='True':
                dict["category_id"] = 1
            else:
                dict["category_id"] = 2

            data.append(dict)

        return data

    def showImage(self, img, anns):
        for ann in anns:
            bbox = ann["bbox"]
            center = ann["center"]
            num = ann["num"]
            category_id = ann["category_id"]

            cv2.line(img, (bbox[0],bbox[1]), (bbox[2],bbox[1]), (0, 255, 0), 1)
            cv2.line(img, (bbox[2], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.line(img, (bbox[2], bbox[3]), (bbox[0], bbox[3]), (0, 255, 0), 1)
            cv2.line(img, (bbox[0], bbox[3]), (bbox[0], bbox[1]), (0, 255, 0), 1)

            cv2.putText(img, num, (center[0], center[1]), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
            if category_id == 1:
                cv2.circle(img, (center[0], center[1]), 3, (0, 0, 255), -1)
            else:
                cv2.circle(img, (center[0], center[1]), 3, (255, 0, 0), -1)

        cv2.imshow("result", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    pano = PANODataset('E:\\dataset\\0210-anonymous\\train')
    file_lists = pano.getImgIds()
    for file in file_lists:
        pano.loadData(file)

