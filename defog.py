# #!/usr/bin/env python
# # encoding: utf-8
#
import os

import cv2
import numpy as np


def haze_removal(image, windowSize=24, w0=0.7, t0=0.1):
    darkImage = image.min(axis=2)  # image(H W C)  darkImage(H, W)
    A = darkImage.max()  # int
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / A)  # t(H, W)
    # T = t * 255  # t(H, W)
    # T.dtype = 'uint8'

    t[t < t0] = t0

    J = image  # image(H W C)
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * A) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * A) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * A) / t
    # result = Image.fromarray(J)
    # return result
    return J


if __name__ == '__main__':

    # import time
    # fps = 0.0
    # img = cv2.imread('00006_3_3_fog.jpg')
    # for i in range(100):
    #     # 字符串拼接
    #     t1 = time.time()
    #     m = haze_removal(img)
    #     fps = (fps + (1. / (time.time() - t1))) / 2  # 计算平均fps
    #     print(i)
    # print(fps)

    ori_dir = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight_trainimages/trainfog"
    dst_dir = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/trainfog"
    ori_imgs = os.listdir(ori_dir)
    tot = len(ori_imgs)
    for idx, name_img in enumerate(ori_imgs):
        ori_path = os.path.join(ori_dir, name_img)
        dst_path = os.path.join(dst_dir, name_img)
        image = cv2.imread(ori_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = haze_removal(image)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(dst_path, result)
        print(idx, "/", tot)
