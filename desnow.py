import os
from cv2 import cv2


def denoise_nl_means(image, h=10, hForColor=10, templateWindowSize=7, searchWindowSize=21):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hForColor, templateWindowSize, searchWindowSize)
    return denoised_image


if __name__ == '__main__':

    # import time
    # fps = 0.0
    # img = cv2.imread("/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight_trainimages/trainsnow/00001.jpg")
    # for i in range(100):
    #     # 字符串拼接
    #     t1 = time.time()
    #     m = denoise_nl_means(img)
    #     fps = (fps + (1. / (time.time() - t1))) / 2  # 计算平均fps
    #     print(i)
    # print(fps)

    # ori_dir = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight_trainimages/trainsnow"
    # dst_dir = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/trainfog"
    # ori_imgs = os.listdir(ori_dir)
    # tot = len(ori_imgs)
    # for idx, name_img in enumerate(ori_imgs):
    #     ori_path = os.path.join(ori_dir, name_img)
    #     dst_path = os.path.join(dst_dir, name_img)
    #     image = cv2.imread(ori_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     result = haze_removal(image)
    #     result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(dst_path, result)
    #     print(idx, "/", tot)

    img = cv2.imread("/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight_trainimages/trainsnow/00440.jpg")
    filtered_image = denoise_nl_means(img)
    cv2.imwrite("./00440.jpg", filtered_image)

