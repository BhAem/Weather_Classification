import cv2
import numpy as np
import time
import os
from os import path

# 伽马变换
def gamma(image, fgamma=0.5):
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    # cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma


if __name__ == '__main__':


    ori_dir = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight_trainimages/trainnight"
    dst_dir = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/trainnight"
    ori_imgs = os.listdir(ori_dir)
    tot = len(ori_imgs)
    for idx, name_img in enumerate(ori_imgs):
        ori_path = os.path.join(ori_dir, name_img)
        dst_path = os.path.join(dst_dir, name_img)
        image = cv2.imread(ori_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = gamma(image)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(dst_path, result)
        print(idx, "/", tot)

    # fps = 0.0
    # # start = time.time()
    # url = 'try_input'
    # file = os.listdir(url)
    # for i in file:
    #     # 字符串拼接
    #     t1 = time.time()
    #     real_url = path.join(url, i)
    #
    #     image = cv2.imread(real_url)
    #
    #     result = gamma(image)
    #     cv2.imwrite('try_output//(' + str(i) + ').jpg', result)
    #     fps = (fps + (1. / (time.time() - t1))) / 2  # 计算平均fps
    # print(fps)