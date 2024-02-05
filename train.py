import os
import argparse

import cv2
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from MWD import MwdDataset, MwdDataset_train
from torch.utils.tensorboard import SummaryWriter
import datetime
from model.resnet import ResNet
from tqdm import tqdm


def train():
    device_ids = [0]
    device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])

    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d %H-%M-%S")
    day = current_time.split()[0].split('-')[2]
    current_time = current_time.split()[1]
    current_time = day + "-" + current_time
    path = "./weights/" + current_time
    if not os.path.exists(path):
        os.makedirs(path)

    train_dataset = MwdDataset_train(opt.root_dir, opt.image_train, (opt.img_size, opt.img_size), True, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=False)
    # test_dataset = MwdDataset(opt.root_dir, opt.image_test, (opt.img_size, opt.img_size), False, transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=False)

    test_day_path = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/test"
    test_rain_path = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/testrain"
    test_snow_path = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/testsnow"
    test_fog_path = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/testfog"
    test_night_path = "/home/Huangwb/Wuqf/yolov5-6.0/TrafficSignLight/images/testnight"

    test_day_imgs = [os.path.join(test_day_path, s) for s in os.listdir(test_day_path)]
    test_rain_imgs = [os.path.join(test_rain_path, s) for s in os.listdir(test_rain_path)]
    test_snow_imgs = [os.path.join(test_snow_path, s) for s in os.listdir(test_snow_path)]
    test_fog_imgs = [os.path.join(test_fog_path, s) for s in os.listdir(test_fog_path)]
    test_night_imgs = [os.path.join(test_night_path, s) for s in os.listdir(test_night_path)]
    test_total = len(test_day_imgs)
    label_idx = {0: "day", 1: "rain", 2: "snow", 3: "fog", 4: "night"}

    net = ResNet()
    if opt.pretrained:
        state_dict = torch.load("/home/Huangwb/Wuqf/classify/resnet18-5c106cde.pth")
        new_state_dict = net.state_dict()
        # 删除pretrained_dict.items()中model所没有的东西
        state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}  # 只保留预训练模型中，自己建的model有的参数
        new_state_dict.update(state_dict)
        net.load_state_dict(new_state_dict)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device=device)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    # total_test_step = 0

    # length 长度
    train_data_size = len(train_dataset)
    # test_data_size = len(test_dataset)
    # 如果train_data_size=10, 训练数据集的长度为：10
    print("训练数据集的长度为：{}".format(train_data_size))
    # print("测试数据集的长度为：{}".format(test_data_size))

    # 添加tensorboard
    writer = SummaryWriter("./logs_train")
    nb_train = len(train_dataloader)  # number of batches
    # nb_test = len(test_dataloader)

    for i in range(opt.epoch):
        print("-------第 {} 轮训练开始-------".format(i+1))
        total_train_step = 0
        total_train_loss = 0
        # 训练步骤开始
        net.train()
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=nb_train)  # progress bar
        optimizer.zero_grad()
        for j, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_step = total_train_step + 1
            # if total_train_step % 10 == 0:
            #     print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            total_train_loss += loss.item()

        writer.add_scalar("train_loss", total_train_loss, i)

        # 测试步骤开始
        net.eval()
        total_test_loss = 0
        total_accuracy = 0
        cnt_day, cnt_rain, cnt_snow, cnt_fog, cnt_night = 0, 0, 0, 0, 0

        with torch.no_grad():
            day_pbar = enumerate(test_day_imgs)
            day_pbar = tqdm(day_pbar, total=test_total)
            for idx, img in day_pbar:
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                output = net(img)
                b = label_idx[int(output.argmax(1))]
                a = "day"
                if a == b:
                    cnt_day += 1

            rain_pbar = enumerate(test_rain_imgs)
            rain_pbar = tqdm(rain_pbar, total=test_total)
            for idx, img in rain_pbar:
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                output = net(img)
                b = label_idx[int(output.argmax(1))]
                a = "rain"
                if a == b:
                    cnt_rain += 1

            snow_pbar = enumerate(test_snow_imgs)
            snow_pbar = tqdm(snow_pbar, total=test_total)
            for idx, img in snow_pbar:
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                output = net(img)
                b = label_idx[int(output.argmax(1))]
                a = "snow"
                if a == b:
                    cnt_snow += 1

            fog_pbar = enumerate(test_fog_imgs)
            fog_pbar = tqdm(fog_pbar, total=test_total)
            for idx, img in fog_pbar:
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                output = net(img)
                b = label_idx[int(output.argmax(1))]
                a = "fog"
                if a == b:
                    cnt_fog += 1

            night_pbar = enumerate(test_night_imgs)
            night_pbar = tqdm(night_pbar, total=test_total)
            for idx, img in night_pbar:
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                output = net(img)
                b = label_idx[int(output.argmax(1))]
                a = "night"
                if a == b:
                    cnt_night += 1

        # with torch.no_grad():
        #     pbar = enumerate(test_dataloader)
        #     pbar = tqdm(pbar, total=nb_test)  # progress bar
        #     for j, (imgs, targets) in pbar:
        #         imgs = imgs.to(device)
        #         targets = targets.to(device)
        #         outputs = net(imgs)
        #         loss = loss_fn(outputs, targets)
        #         total_test_loss = total_test_loss + loss.item()
        #         accuracy = (outputs.argmax(1) == targets).sum()
        #         total_accuracy = total_accuracy + accuracy

        # acc = total_accuracy/test_data_size
        # print_string = "epoch {:>3} | train_loss: {:.5f} | test_loss: {:.5f} | acc: {:.5f}"
        # print(print_string.format(i, total_train_loss, total_test_loss, acc))

        acc_day, acc_rain, acc_snow, acc_fog, acc_night = cnt_day/test_total, cnt_rain/test_total, \
                                            cnt_snow/test_total, cnt_fog/test_total, cnt_night/test_total
        print_string = "epoch {:>3} | train_loss: {:.5f} | test_loss: {:.5f} | acc_day: {:.5f} | acc_rain: {:.5f} | acc_snow: {:.5f} | acc_fog: {:.5f} | acc_night: {:.5f}"
        print(print_string.format(i, total_train_loss, total_test_loss, acc_day, acc_rain, acc_snow, acc_fog, acc_night))

        # print("整体测试集上的Loss: {}".format(total_test_loss))
        # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
        # writer.add_scalar("test_loss", total_test_loss, i)
        # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, i)
        # total_test_step = total_test_step + 1

        # torch.save(net.state_dict(), path + "/net_{}_{:.4f}_{:.4f}.pth".format(i, acc, total_test_loss))
        torch.save(net.state_dict(), path + "/net_{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.pth".format(i, acc_day, acc_rain, acc_snow, acc_fog, acc_night))
        # print("模型已保存")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./MWD")
    parser.add_argument('--image_train', type=str, default="train")
    parser.add_argument('--image_test', type=str, default="test_old")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--pretrained', type=bool, default=True)
    # parser.add_argument('--att', type=str, default='TA', choices=[None, 'CBAM', 'BAM', 'scSE', 'simam', 'TA', 'CA'])
    opt = parser.parse_args()
    train()

    # python train.py > train.log 2>&1 &
    # tail -f -n100 train.log

