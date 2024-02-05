import os
import time
import numpy as np
import torch
import cv2
import torchvision.models
from torchvision import transforms
from tqdm import tqdm

from model.resnet import ResNet


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
              stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


label_idx = {0: "day", 1: "rain", 2: "snow", 3: "fog", 4: "night"}
device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
image_path = "/home/Huangwb/Wuqf/classify/MWD/test3"

imgs = [s for s in os.listdir(image_path)]
model = ResNet().to(device)
# ckpt = torch.load("/home/Huangwb/Wuqf/classify/weights/22-17-34-12/net_5_0.9934_2.7148.pth", map_location=device)
# ckpt = torch.load("/home/Huangwb/Wuqf/classify/weights/22-19-43-22/net_6_0.9966_2.0305.pth", map_location=device)
# ckpt = torch.load("/home/Huangwb/Wuqf/classify/weights/23-22-40-29/net_12_0.9946_2.2858.pth", map_location=device)
ckpt = torch.load("/home/Huangwb/Wuqf/classify/weights/25-01-33-56/net_4_1.0_1.0_1.0_1.0_1.0.pth", map_location=device)
model.load_state_dict(ckpt)
model.eval()
fps = 0.0  # 计算帧数
cnt = 0
total = len(imgs)
pbar = enumerate(imgs)
pbar = tqdm(pbar, total=total)
with torch.no_grad():
    for idx, img_name in pbar:
        img = os.path.join(image_path, img_name)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # h0, w0 = img.shape[:2]  # orig hw
        # r = 640 / max(h0, w0)  # ratio
        # if r != 1:  # if sizes are not equal
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        # img, ratio, pad = letterbox(img, new_shape=640, auto=False, scaleup=False)

        img = transforms.ToTensor()(img).unsqueeze(0)
        img = img.to(device)
        torch.cuda.synchronize()
        t1 = time.time()
        output = model(img)
        torch.cuda.synchronize()
        t2 = time.time()
        fps = (fps + (1. / (t2 - t1 + 0.0000000001))) / 2  # 计算平均fps
        b = label_idx[int(output.argmax(1))]
        # a = img_name.split(".")[0].split("_")[-1]
        a = "day"
        if a == b:
            cnt += 1
        # print(str(idx) + "/" + str(total), img.shape)

print(cnt / total)
print(fps)


