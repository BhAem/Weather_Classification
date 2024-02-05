import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from PIL import Image
from torchvision import transforms
import random

# writer = SummaryWriter("logs")

class MwdDataset_train(Dataset):

    def __init__(self, root_dir, image_dir, image_size, is_train, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.is_train = is_train
        self.transform = transform
        self.label_idx = {'day': 0, 'rain': 1, 'snow': 2, 'fog': 3, 'night': 4}
        self.num_seed = 0

        # try:
        #     self.brightness = (0.8, 1.2)
        #     self.contrast = (0.8, 1.2)
        #     self.saturation = (0.8, 1.2)
        #     self.hue = (-0.1, 0.1)
        #     self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        # except TypeError:
        #     self.brightness = 0.2
        #     self.contrast = 0.2
        #     self.saturation = 0.2
        #     self.hue = 0.1

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        self.num_seed = random.randint(0, 4)
        if self.num_seed == 0:
            label = 'day'
            image_dir = self.image_dir
        elif self.num_seed == 1:
            label = 'rain'
            image_dir = self.image_dir + 'rain'
        elif self.num_seed == 2:
            label = 'snow'
            image_dir = self.image_dir + 'snow'
        elif self.num_seed == 3:
            label = 'fog'
            image_dir = self.image_dir + 'fog'
            img_name = img_name.split('.')[0] + "_fog" + ".jpg"
        elif self.num_seed == 4:
            label = 'night'
            image_dir = self.image_dir + 'night'
            img_name = img_name.split('.')[0] + "_night" + ".jpg"
        # self.num_seed += 1
        # self.num_seed %= 6
        img_item_path = os.path.join(self.root_dir, image_dir, img_name)

        # label = img_name.split('.')[0].split('_')[-1]
        # img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)

        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(img_item_path).convert('RGB')
        img = cv2.resize(img, self.image_size)
        # img = img.resize(self.image_size)
        # h0, w0 = img.shape[:2]  # orig hw
        # r = 640 / max(h0, w0)  # ratio
        # if r != 1:  # if sizes are not equal
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        # img, ratio, pad = self.letterbox(img, 640, auto=False, scaleup=False)

        # if self.is_train and random.random() > 0.5:
        #     img = transforms.RandomRotation(180)(img)
        #
        # if self.is_train:
        #     img = transforms.RandomHorizontalFlip()(img)
        #
        # if self.is_train:
        #     img = transforms.RandomVerticalFlip()(img)
        #
        # # 颜色增强参数设定
        # if self.is_train and random.random() > 0.5:
        #     color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        #     img = self.color_aug(img)
        #
        # if self.is_train and random.random() > 0.5:
        #     shape = img.size
        #     shape = (shape[1], shape[0])
        #     img = transforms.RandomResizedCrop(shape)(img)
        #
        # if self.is_train and random.random() > 0.5:
        #     img = transforms.GaussianBlur(kernel_size=(5, 11), sigma=(5, 10.0))(img)

        # img = np.asarray(img)

        if self.transform:
            # img = img / 255.0
            img = self.transform(img)

        return img, self.label_idx[label]

    def __len__(self):
        return len(self.image_list)


class MwdDataset(Dataset):

    def __init__(self, root_dir, image_dir, image_size, is_train, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.is_train = is_train
        self.transform = transform
        self.label_idx = {'day': 0, 'rain': 1, 'snow': 2, 'fog': 3, 'night': 4}

        # try:
        #     self.brightness = (0.8, 1.2)
        #     self.contrast = (0.8, 1.2)
        #     self.saturation = (0.8, 1.2)
        #     self.hue = (-0.1, 0.1)
        #     self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        # except TypeError:
        #     self.brightness = 0.2
        #     self.contrast = 0.2
        #     self.saturation = 0.2
        #     self.hue = 0.1

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = img_name.split('.')[0].split('_')[-1]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)

        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(img_item_path).convert('RGB')
        img = cv2.resize(img, self.image_size)
        # img = img.resize(self.image_size)
        # h0, w0 = img.shape[:2]  # orig hw
        # r = 640 / max(h0, w0)  # ratio
        # if r != 1:  # if sizes are not equal
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        # img, ratio, pad = self.letterbox(img, 640, auto=False, scaleup=False)
        # if self.is_train and random.random() > 0.5:
        #     img = transforms.RandomRotation(180)(img)
        #
        # if self.is_train:
        #     img = transforms.RandomHorizontalFlip()(img)
        #
        # if self.is_train:
        #     img = transforms.RandomVerticalFlip()(img)
        #
        # # 颜色增强参数设定
        # if self.is_train and random.random() > 0.5:
        #     color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        #     img = self.color_aug(img)
        #
        # if self.is_train and random.random() > 0.5:
        #     shape = img.size
        #     shape = (shape[1], shape[0])
        #     img = transforms.RandomResizedCrop(shape)(img)
        #
        # if self.is_train and random.random() > 0.5:
        #     img = transforms.GaussianBlur(kernel_size=(5, 11), sigma=(5, 10.0))(img)

        # img = np.asarray(img)

        if self.transform:
            # img = img / 255.0
            img = self.transform(img)

        return img, self.label_idx[label]

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    root_dir = "./MWD"
    image_train = "train"
    train_dataset = MwdDataset(root_dir, image_train, (224, 224), transform)

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1)

    # writer.add_image('error', train_dataset[119]['img'])
    # writer.close()
    for i, j in enumerate(dataloader):
        # imgs, labels = j
        print(i, j[0].shape, j[1])
        # writer.add_image("train_data_b2", make_grid(j['img']), i)
    #
    # writer.close()
