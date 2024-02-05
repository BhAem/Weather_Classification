import os
import random
import shutil


def data_set_split(src_img_folder, target_train_img_folder, target_test_img_folder, train_scale=0.8, test_scale=0.2):

    print("开始数据集划分")
    imgs = [s for s in os.listdir(src_img_folder)]
    current_data_length = len(imgs)
    current_data_index_list = list(range(current_data_length))

    # current_data_index_list = current_data_index_list[0:int(current_data_length*0.44)]
    # current_data_length = int(current_data_length*0.44)   # 0.58 cloudy  0.44 sunny

    random.shuffle(current_data_index_list)
    # print(current_data_index_list)
    train_stop_flag = int(current_data_length * train_scale)
    # test_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    test_num = 0
    for i in current_data_index_list:
        if current_idx <= train_stop_flag:
            src_img = os.path.join(src_img_folder, imgs[i])
            dst_img = os.path.join(target_train_img_folder, imgs[i])
        else:
            src_img = os.path.join(src_img_folder, imgs[i])
            dst_img = os.path.join(target_test_img_folder, imgs[i])
        shutil.copy(src_img, dst_img)

        # if current_idx <= train_stop_flag:
        #     shutil.copy(src_img, target_train_img_folder)
        #     shutil.copy(src_txt, target_train_label_folder)
        #     # print("{}复制到了{}".format(src_img_path, train_folder))
        #     train_num = train_num + 1
        # else:
        #     shutil.copy(src_img, target_test_img_folder)
        #     shutil.copy(src_txt, target_test_label_folder)
        #     # print("{}复制到了{}".format(src_img_path, test_folder))
        #     test_num = test_num + 1

        current_idx = current_idx + 1

    imgs = [s.split('.')[0] for s in os.listdir(target_train_img_folder)]
    imgs1 = [s.split('.')[0] for s in os.listdir(target_test_img_folder)]
    print(len(imgs))
    print(len(imgs1))

    # txts = [s.split('.')[0] for s in os.listdir(target_train_label_folder)]
    # txts1 = [s.split('.')[0] for s in os.listdir(target_test_label_folder)]
    # print(len(txts))
    # print(len(txts1))


if __name__ == '__main__':
    src_img_folder = "D:\\Users\\11939\\Desktop\\c_traffic\\MWD\\haze"
    # src_label_folder = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data\\labels\\test"
    target_train_img_folder = "D:\\Users\\11939\\Desktop\\c_traffic\\MWD\\train"
    # target_train_label_folder = "D:\\winter\\UA_DETRAC_yolo_data3\\labels\\train\\"
    target_test_img_folder = "D:\\Users\\11939\\Desktop\\c_traffic\\MWD\\test"
    # target_test_label_folder = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\total2\\labels"
    data_set_split(src_img_folder, target_train_img_folder, target_test_img_folder)