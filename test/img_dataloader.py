import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import time
from img_loader import img_loader
import csv

# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    datasetReader = img_loader(r'C:\Users\fuchi\Desktop\working\abc')
    dataloader = DataLoader(datasetReader, batch_size=512, shuffle=False, pin_memory=True, num_workers=12)

    num_vec = 0
    dim_vec = 4096
    detections_batch = []
    for i, (batch) in enumerate(dataloader):
        num_vec += len(batch[1])

    hash_cal_end = time.time()
    print(f"FPS per sec {num_vec / (hash_cal_end - start)}")
    print("Hash calculation time: ", hash_cal_end - start)
