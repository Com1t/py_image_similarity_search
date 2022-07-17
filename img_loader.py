import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

torch.manual_seed(1)  # reproducible


class img_loader(Dataset):
    def __init__(self, datapath):
        self.x_datalist = []
        self.y_datalist = []
        self.datapath = datapath
        self.img_extensions = ['jpeg', 'psd', 'jpg', 'png', 'gif']
        for file in os.listdir(datapath):
            filename = os.fsdecode(file)
            if filename.split('.')[-1].lower() in self.img_extensions:
                self.y_datalist.append(filename)

        # transform
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        label = self.y_datalist[index]
        image = Image.open(os.path.join(self.datapath, label)).convert('RGB')
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.y_datalist)
