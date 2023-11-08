import os
import cv2
import torch
import pandas
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler


class MyDataset(Dataset):
    def __init__(self, annotation, root, transform=None, is_dog_cat=False, last_cat = 11):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.is_dog_cat = is_dog_cat
        self.last_cat = last_cat

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        if not self.is_dog_cat:
            landmarks = self.landmarks_frame.iloc[idx, 1]
        else:
            landmarks = self.landmarks_frame.iloc[idx, 2]
        classification_labels = int(landmarks)
        return image, classification_labels

    def getSampler(self, name_clases="binaryclass"):
        rasp_sampler_list = []
        class_count = self.landmarks_frame[name_clases].value_counts()
        rasp_sampler_list = [
            1 / class_count[i] for i in self.landmarks_frame[name_clases].values
        ]

        if self.__len__() != len(rasp_sampler_list):
            raise ValueError("sampler does not converge with the map")
        sampler = WeightedRandomSampler(rasp_sampler_list, len(rasp_sampler_list))
        return sampler

class MyDataset_gen(Dataset):
    def __init__(self, annotation, root, transform=None, is_dog_cat=False, last_cat = 11, transforms=[]):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.is_dog_cat = is_dog_cat
        self.last_cat = last_cat

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image_t = self.transform(image=image)["image"]
        images = []
        if len(self.transforms):
            for t in self.transforms:
                images.append(t(image=image)["image"])
        
        return image_t, images

    def getSampler(self, name_clases="binaryclass"):
        rasp_sampler_list = []
        class_count = self.landmarks_frame[name_clases].value_counts()
        rasp_sampler_list = [
            1 / class_count[i] for i in self.landmarks_frame[name_clases].values
        ]

        if self.__len__() != len(rasp_sampler_list):
            raise ValueError("sampler does not converge with the map")
        sampler = WeightedRandomSampler(rasp_sampler_list, len(rasp_sampler_list))
        return sampler