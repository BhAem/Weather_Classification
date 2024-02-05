import cv2
import numpy as np
import time
import os
from os import path

import torch


# 伽马变换
def gamma(image, fgamma=0.5):
    image_gamma = torch.pow(image, fgamma)
    return image_gamma


if __name__ == '__main__':

    imageName = '00011_night.jpg'
    image = cv2.imread("./" + imageName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    device_ids = [0]
    device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
    y = np.float32(image) / 255.
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = torch.Tensor(y)
    image = y.to(device)

    imageName2 = '00011_night.jpg'
    image2 = cv2.imread("./" + imageName2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    y = np.float32(image2) / 255.
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = torch.Tensor(y)
    image2 = y.to(device)

    image = torch.cat([image, image2], dim=0)

    result2 = gamma(image, 0.5)

    result = result2[0]
    save_out = np.uint8(255 * result.data.cpu().numpy().squeeze())
    save_out = save_out.transpose(1, 2, 0)
    save_out = cv2.cvtColor(save_out, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./00011_2.jpg', save_out)

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