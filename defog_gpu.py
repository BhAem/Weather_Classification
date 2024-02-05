# #!/usr/bin/env python
# # encoding: utf-8
#
import cv2
import numpy as np
import torch


def haze_removal(image, w0=0.7, t0=0.1):

    darkImage, _ = torch.min(image, dim=1)  # image(b,c,h,w)  darkImage(b,h,w)
    A, _ = torch.max(torch.max(darkImage, dim=1, keepdim=True)[0], dim=2, keepdim=True)  # A(b,1,1)
    t = 1 - w0 * (darkImage / A)
    t[t < t0] = t0
    J = image
    J[:, 0, :, :] = (image[:, 0, :, :] - (1 - t) * A) / t
    J[:, 1, :, :] = (image[:, 1, :, :] - (1 - t) * A) / t
    J[:, 2, :, :] = (image[:, 2, :, :] - (1 - t) * A) / t
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

    imageName = "99.png"
    image = cv2.imread("./" + imageName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    device_ids = [0]
    device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
    y = np.float32(image) / 255.
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = torch.Tensor(y)
    image = y.to(device)

    imageName2 = "99.png"
    image2 = cv2.imread("./" + imageName2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    y = np.float32(image2) / 255.
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = torch.Tensor(y)
    image2 = y.to(device)

    image = torch.cat([image, image2], dim=0)

    result2 = haze_removal(image)
    result = result2[0]

    save_out = np.uint8(255 * result.data.cpu().numpy().squeeze())
    save_out = save_out.transpose(1, 2, 0)
    save_out = cv2.cvtColor(save_out, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./99_1.png', save_out)