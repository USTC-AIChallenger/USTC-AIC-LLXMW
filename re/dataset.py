import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE
import numpy as np

class Random_erase:
    def __init__(self, p, area_ratio_range, min_aspect_ratio, max_attempt):
        self.p = p
        self.sl, self.sh = area_ratio_range
        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio
        self.max_attempt = max_attempt

    def __call__(self, image):
        image = np.array(image)
#	print image
#        print(image.shape)
        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            if np.random.random() > self.p:
                mask_area = np.random.uniform(self.sl, self.sh) * image_area
                aspect_ratio = np.random.uniform(self.rl, self.rh)
                mask_h = int(np.sqrt(mask_area * aspect_ratio))
                mask_w = int(np.sqrt(mask_area / aspect_ratio))

                if mask_w < w and mask_h < h:
                    x0 = np.random.randint(0, w - mask_w)
                    y0 = np.random.randint(0, h - mask_h)
                    x1 = x0 + mask_w
                    y1 = y0 + mask_h
                    image[y0:y1, x0:x1] = np.random.randint(0, 255) #np.random.randint(0,100)#np.random.uniform(0, 1)
        image = Image.fromarray(np.uint8(image))
        return image

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = Random_erase(0.5,[0.02,0.2], 0.3,2)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    INPUT_SIZE = (448, 448)
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
