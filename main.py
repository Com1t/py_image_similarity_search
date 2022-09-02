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
    model = models.alexnet(pretrained=True).eval().to(device)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(6)])

    start = time.time()
    datasetReader = img_loader(r'C:\Users\fuchi\Desktop\abc')
    dataloader = DataLoader(datasetReader, batch_size=512, shuffle=False, pin_memory=True, num_workers=12)

    num_vec = 0
    dim_vec = 4096
    detections_batch = []
    for i, (batch) in enumerate(dataloader):
        num_vec += len(batch[1])
        data_for_process = batch[0].to(device)
        print(data_for_process.shape)
        result = model(data_for_process).detach().cpu().numpy()
        detections_batch.append([result, batch[1]])

    hash_cal_end = time.time()
    print("Hash calculation time: ", hash_cal_end - start)

    vec_header = ['filename']
    [vec_header.append(f'vec_{i}') for i in range(dim_vec)]

    with open(rf'/Users/fuchiang137/Downloads/image_vec_n_{num_vec}_d_{dim_vec}.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow([num_vec, dim_vec])
        writer.writerow(vec_header)
        for res_batch in detections_batch:
            for i in range(res_batch[0].shape[0]):
                row = [res_batch[1][i]]
                [row.append(j) for j in res_batch[0][i]]
                writer.writerow(row)

    end = time.time()

    print(f'write file time: {end - hash_cal_end}')

    print(f'hash total time: {end - start}')
